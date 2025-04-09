import torch
import logging
from .utils import is_numeric, get_regular_embeddings

def evaluate_fne(model, test_loader, number_encoder, intermediate_network, int_digit_len, frac_digit_len, device, print_labels=False, max_print=10, decoder_type='fourier', tokenizer=None):
    """
    Evaluation loop for Fourier Neural Embedding (FNE) based models.
    
    Parameters:
        decoder_type: 'fourier' (default) or 'greedy'
        tokenizer: Required when decoder_type is 'greedy'
    """
    logging.info('Evaluation start')
    model.eval()
    number_encoder.eval()
    intermediate_network.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    total_squared_error = 0
    total_digits = 0
    correct_digits = 0
    all_labels = []
    all_predictions = []
    mispredictions = []

    if decoder_type == 'greedy' and tokenizer is None:
        raise ValueError("Tokenizer is required when decoder_type is 'greedy'")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            scatter_tensor = batch['scatter_tensor'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            last_token_mask = batch['last_token_mask'].to(device)

            regular_embeddings = get_regular_embeddings(model, input_ids)
            fourier_embeddings = number_encoder(scatter_tensor)
            fourier_embeddings = intermediate_network(fourier_embeddings)
            input_embeddings = regular_embeddings + fourier_embeddings

            # modified part by EJ: match dtype of inputs same as model dtype
            input_embeddings = input_embeddings.to(model.dtype)
            attention_mask = attention_mask.to(model.dtype)     

            if decoder_type == 'fourier':
                outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask, output_hidden_states=True)
                before_decoder = outputs.hidden_states[-1]
                last_token_hidden_state = (before_decoder * last_token_mask.unsqueeze(-1)).sum(dim=1)
                
                predicted_numbers = number_encoder.fourier_compute_prediction(last_token_hidden_state, int_digit_len, frac_digit_len)
                
                all_labels.append(labels.cpu())
                all_predictions.append(predicted_numbers.cpu())
                
                tolerance = 10 ** (-frac_digit_len)
                correct_predictions = torch.abs(predicted_numbers - labels) < tolerance
                total_correct += correct_predictions.sum().item()
                total_samples += labels.size(0)

                for i in range(labels.size(0)):
                    actual_value = str(labels[i].item())
                    predicted_value = str(predicted_numbers[i].item())
                    min_len = len(actual_value)
                    correct_digits += sum(1 for a, p in zip(actual_value[:min_len], predicted_value[:min_len]) if a == p)
                    total_digits += len(actual_value)

                for i in range(labels.size(0)):
                    if not correct_predictions[i]:
                        mispredictions.append((predicted_numbers[i].item(), labels[i].item()))

                squared_error = torch.sum((predicted_numbers - labels) ** 2).item()
                total_squared_error += squared_error

                loss = number_encoder.fourier_compute_loss(last_token_hidden_state, labels, int_digit_len, frac_digit_len)
                total_loss += loss.item()
            
            elif decoder_type == 'greedy':
                # Use the greedy decoding method from evaluate_regular
                outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Get predictions using argmax
                predictions = torch.argmax(logits, dim=-1)
                
                batch_correct = 0
                batch_total = len(input_ids)
                
                for i in range(len(input_ids)):
                    label_indices = (batch['labels'][i] != -100).nonzero(as_tuple=True)[0]
                    actual_tokens = input_ids[i, label_indices].cpu().numpy()
                    predicted_tokens = predictions[i, label_indices-1].cpu().numpy()
                    
                    actual_label = tokenizer.decode(actual_tokens, skip_special_tokens=True).strip()
                    predicted_label = tokenizer.decode(predicted_tokens, skip_special_tokens=True).strip()
                    
                    # Convert to float for comparison
                    if is_numeric(predicted_label) and is_numeric(actual_label):
                        actual_value = float(actual_label)
                        predicted_value = float(predicted_label)
                        
                        # Add to metrics
                        if abs(predicted_value - actual_value) < 10 ** (-frac_digit_len):
                            batch_correct += 1
                        
                        # Store for MSE calculation
                        squared_error = (predicted_value - actual_value) ** 2
                        total_squared_error += squared_error
                        
                        # For mispredictions
                        if abs(predicted_value - actual_value) >= 10 ** (-frac_digit_len):
                            mispredictions.append((predicted_value, actual_value))
                            
                        # Character-wise accuracy
                        str_actual = str(actual_value)
                        str_predicted = str(predicted_value)
                        min_len = min(len(str_actual), len(str_predicted))
                        correct_digits += sum(1 for a, p in zip(str_actual[:min_len], str_predicted[:min_len]) if a == p)
                        total_digits += len(str_actual)
                        
                        # Collecting all labels and predictions for R2 calculation
                        all_labels.append(actual_value)
                        all_predictions.append(predicted_value)
                
                total_correct += batch_correct
                total_samples += batch_total
                
                # Pseudo-loss for consistency in return values
                total_loss += 0  # We don't have a proper loss here

    # Calculate metrics based on decoder type
    if decoder_type == 'fourier':
        all_labels = torch.cat(all_labels)
        mean_label = all_labels.mean().item()
        total_variance = torch.sum((all_labels - mean_label) ** 2).item()
    else:  # greedy
        if all_labels:
            mean_label = sum(all_labels) / len(all_labels)
            total_variance = sum((label - mean_label) ** 2 for label in all_labels)
        else:
            mean_label = 0
            total_variance = 0
            raise ValueError("No labels found for greedy decoder")

    avg_loss = total_loss / len(test_loader)
    whole_number_accuracy = total_correct / total_samples
    digit_wise_accuracy = correct_digits / total_digits if total_digits > 0 else 0
    mse = total_squared_error / total_samples
    r2 = 1 - (total_squared_error / total_variance) if total_variance > 0 else float('nan')

    if print_labels:
        if mispredictions:
            log_count = min(len(mispredictions), max_print)
            logging.info(f"Mispredictions (up to {log_count} examples):")
            for i in range(log_count):
                predicted_val, actual_val = mispredictions[i]
                logging.info(f"Predicted: {predicted_val}, Actual: {actual_val}")
        else:
            logging.info("No mispredictions found!")

    return avg_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2

def evaluate_regular(model, dataloader, tokenizer, device, print_labels=False, max_print_examples=10):
    """
    Evaluation loop for regular models.
    """
    logging.info('Evaluation start')
    model.eval()
    total_loss = 0
    total_examples = 0
    total_correct_examples = 0
    total_characters = 0
    correct_characters = 0
    total_squared_error = 0
    all_labels = []
    printed_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=-1)
            examplelist = []
            for i in range(len(input_ids)):
                label_indices = (labels[i] != -100).nonzero(as_tuple=True)[0]
                actual_tokens = input_ids[i, label_indices].cpu().numpy()
                predicted_tokens = predictions[i, label_indices-1].cpu().numpy()
                actual_label = tokenizer.decode(actual_tokens, skip_special_tokens=True).strip()
                predicted_label = tokenizer.decode(predicted_tokens, skip_special_tokens=True).strip()

                if actual_label == predicted_label:
                    total_correct_examples += 1
                total_examples += 1

                if is_numeric(predicted_label):
                    actual_value = float(actual_label)
                    predicted_value = float(predicted_label)
                    total_squared_error += (actual_value - predicted_value) ** 2
                    all_labels.append(actual_value)

                max_len = max(len(actual_label), len(predicted_label))
                padded_actual = actual_label.ljust(max_len)
                padded_predicted = predicted_label.ljust(max_len)
                
                correct_characters += sum(1 for a, p in zip(padded_actual, padded_predicted) if a == p)
                total_characters += max_len

                if print_labels and printed_examples < max_print_examples:
                    examplelist.append(f"({predicted_label}, {actual_label})")
                    printed_examples += 1

            if print_labels and examplelist:
                logging.info(" ".join(examplelist))

    avg_loss = total_loss / len(dataloader)
    whole_number_accuracy = total_correct_examples / total_examples
    digit_wise_accuracy = correct_characters / total_characters

    if all_labels:
        mean_label = sum(all_labels) / len(all_labels)
        total_variance = sum((label - mean_label) ** 2 for label in all_labels)
        mse = total_squared_error / len(all_labels)
        r2 = 1 - (total_squared_error / total_variance) if total_variance > 0 else float('nan')
    else:
        mse = -1
        r2 = float('nan')

    return avg_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2

def evaluate_xval(model, test_loader, xval, device, print_labels=False, max_print=10):
    """
    Evaluation loop for models using the xval module.
    """
    logging.info('Evaluation start')
    model.eval()
    xval.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    total_squared_error = 0
    total_digits = 0
    correct_digits = 0
    printed_examples = 0
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            scatter_tensor = batch['scatter_tensor'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            last_token_mask = batch['last_token_mask'].to(device)

            regular_embeddings = get_regular_embeddings(model, input_ids)
            input_embeddings = xval(scatter_tensor, regular_embeddings)

            outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask, output_hidden_states=True)
            before_decoder = outputs.hidden_states[-1]
            last_token_hidden_state = (before_decoder * last_token_mask.unsqueeze(-1)).sum(dim=1)

            predicted_numbers = xval.compute_prediction(last_token_hidden_state)
            
            tolerance = 0.5  # Example tolerance value
            correct_predictions = torch.abs(predicted_numbers - labels) < tolerance
            
            total_correct += correct_predictions.sum().item()
            total_samples += labels.size(0)
            all_labels.extend(labels.cpu().numpy())

            for i in range(labels.size(0)):
                actual_value = str(labels[i].item())
                predicted_value = str(predicted_numbers[i].item())
                min_len = len(actual_value)
                correct_digits += sum(1 for a, p in zip(actual_value[:min_len], predicted_value[:min_len]) if a == p)
                total_digits += len(actual_value)

            loss = xval.compute_loss(last_token_hidden_state, labels)
            total_loss += loss.item()
            total_squared_error += torch.sum((predicted_numbers - labels) ** 2).item()

            if print_labels and printed_examples < max_print:
                output_pairs = []
                for i in range(len(labels)):
                    if printed_examples >= max_print:
                        break
                    actual_label = labels[i].cpu().numpy()
                    predicted_label = predicted_numbers[i].cpu().numpy()
                    output_pairs.append((predicted_label, actual_label))
                    printed_examples += 1
                logging.info("Predictions and Labels: " + " ".join(f"({pred},{lbl})" for pred, lbl in output_pairs))

    avg_loss = total_loss / len(test_loader)
    whole_number_accuracy = total_correct / total_samples
    digit_wise_accuracy = correct_digits / total_digits
    mse = total_squared_error / total_samples

    if total_samples > 1:
        mean_label = sum(all_labels) / len(all_labels)
        total_variance = sum((label - mean_label) ** 2 for label in all_labels)
        r2 = 1 - (total_squared_error / total_variance) if total_variance > 0 else float('nan')
    else:
        r2 = float('nan')

    return avg_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2

def evaluate_vanilla(model, test_loader, vanilla_model, intermediate_network, device, print_labels=False, max_print=10):
    """
    Evaluation loop for models using the vanilla embedding module.
    """
    model.eval()
    vanilla_model.eval()
    intermediate_network.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    total_squared_error = 0
    total_digits = 0
    correct_digits = 0
    mispredictions = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            scatter_tensor = batch['scatter_tensor'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            last_token_mask = batch['last_token_mask'].to(device)

            regular_embeddings = get_regular_embeddings(model, input_ids)
            vanilla_embeddings = vanilla_model(scatter_tensor)
            vanilla_embeddings = intermediate_network(vanilla_embeddings)
            input_embeddings = regular_embeddings + vanilla_embeddings
            
            # Match dtype of inputs with model
            input_embeddings = input_embeddings.to(model.dtype)
            attention_mask = attention_mask.to(model.dtype)

            outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            last_token_hidden_state = (last_hidden_state * last_token_mask.unsqueeze(-1)).sum(dim=1)

            predicted_numbers = vanilla_model.compute_prediction(last_token_hidden_state)
            
            tolerance = 10 ** (-vanilla_model.frac_digit_len)
            correct = torch.abs(predicted_numbers - labels) < tolerance
            total_correct += correct.sum().item()
            total_samples += labels.size(0)
            
            scaled_labels = (labels * (10 ** vanilla_model.frac_digit_len)).long()
            scaled_preds = (predicted_numbers * (10 ** vanilla_model.frac_digit_len)).long()
            
            for i in range(labels.size(0)):
                label_digits = []
                pred_digits = []
                # Extract digits from label
                num = scaled_labels[i]
                for p in vanilla_model.powers_of_ten:
                    label_digits.append((num // p) % 10)
                # Extract digits from prediction
                num = scaled_preds[i]
                for p in vanilla_model.powers_of_ten:
                    pred_digits.append((num // p) % 10)
                # Compare digit-by-digit
                for l, p in zip(label_digits, pred_digits):
                    if l == p:
                        correct_digits += 1
                    total_digits += 1

            for i in range(labels.size(0)):
                if not correct[i]:
                    mispredictions.append((predicted_numbers[i].item(), labels[i].item()))

            loss = vanilla_model.compute_loss(last_token_hidden_state, labels)
            total_loss += loss.item()
            total_squared_error += torch.sum((predicted_numbers - labels) ** 2).item()

    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_samples
    digit_accuracy = correct_digits / total_digits
    mse = total_squared_error / total_samples
    all_labels_tensor = torch.cat([batch['labels'].to(device) for batch in test_loader])
    mean_label = all_labels_tensor.mean()
    total_variance = torch.sum((all_labels_tensor - mean_label) ** 2).item()
    r2 = 1 - (total_squared_error / total_variance) if total_variance != 0 else 0

    if print_labels and mispredictions:
        logging.info(f"Mispredictions (first {max_print}):")
        for pred, true in mispredictions[:max_print]:
            logging.info(f"Predicted: {pred:.5f}, True: {true:.5f}")

    return avg_loss, (accuracy, digit_accuracy), mse, r2
