import torch
import logging
import sys
from train.utils import get_regular_embeddings, handle_nan_loss

def train_fne(model, train_loader, number_encoder, intermediate_network, optimizer, scheduler, args, int_digit_len, frac_digit_len, len_gen_size, decoder_type, adapter_type, device, tokenizer=None):
    """
    Training loop for Fourier Neural Embedding (FNE) based models with intermediate network.
    LLM parameters are now trainable along with FNE and intermediate network.
    
    Parameters:
        tokenizer: Required when decoder_type is 'greedy'
    """
    # If using greedy decoder, delegate to train_vanilla
    if decoder_type == 'greedy':
        logging.info("Using vanilla training method for greedy decoder")
        
        # Create a simple adapter function for the compute_loss method
        orig_compute_loss = number_encoder.fourier_compute_loss
        orig_compute_prediction = number_encoder.fourier_compute_prediction
        
        # Monkey patch the methods temporarily
        def patched_compute_loss(self, last_hidden_state, label):
            return orig_compute_loss(last_hidden_state, label, int_digit_len, frac_digit_len)
            
        def patched_compute_prediction(self, last_hidden_state):
            return orig_compute_prediction(last_hidden_state, int_digit_len, frac_digit_len)
        
        # Save original methods and attributes to restore later
        number_encoder._original_compute_loss = getattr(number_encoder, 'compute_loss', None)
        number_encoder._original_compute_prediction = getattr(number_encoder, 'compute_prediction', None)
        number_encoder._original_frac_digit_len = getattr(number_encoder, 'frac_digit_len', None)
        number_encoder._original_int_digit_len = getattr(number_encoder, 'int_digit_len', None)
        
        # Add required attributes for vanilla training
        number_encoder.frac_digit_len = frac_digit_len  # Add this attribute directly
        number_encoder.int_digit_len = int_digit_len    # Also add int_digit_len for completeness
        
        # Add the patched methods
        import types
        number_encoder.compute_loss = types.MethodType(patched_compute_loss, number_encoder)
        number_encoder.compute_prediction = types.MethodType(patched_compute_prediction, number_encoder)
        
        # try:
        #     # Run training with patched encoder
        #     # model, dataloader, optimizer, scheduler, device, args# 
        #     return train_regular(model, train_loader, optimizer, scheduler, device, args, number_encoder, intermediate_network)
        # finally:
        #     # Restore original methods
        #     if number_encoder._original_compute_loss is not None:
        #         number_encoder.compute_loss = number_encoder._original_compute_loss
        #     else:
        #         delattr(number_encoder, 'compute_loss')
                
        #     if number_encoder._original_compute_prediction is not None:
        #         number_encoder.compute_prediction = number_encoder._original_compute_prediction
        #     else:
        #         delattr(number_encoder, 'compute_prediction')
                
        #     # Restore original attributes (or remove them if they didn't exist)
        #     if number_encoder._original_frac_digit_len is not None:
        #         number_encoder.frac_digit_len = number_encoder._original_frac_digit_len
        #     else:
        #         delattr(number_encoder, 'frac_digit_len')
                
        #     if number_encoder._original_int_digit_len is not None:
        #         number_encoder.int_digit_len = number_encoder._original_int_digit_len
        #     else:
        #         delattr(number_encoder, 'int_digit_len')
                
        #     # Clean up temporary attributes
        #     delattr(number_encoder, '_original_compute_loss')
        #     delattr(number_encoder, '_original_compute_prediction')
        #     delattr(number_encoder, '_original_frac_digit_len')
        #     delattr(number_encoder, '_original_int_digit_len')
    
    # Ensure the tokenizer is provided when using greedy decoder
    if decoder_type == 'greedy' and tokenizer is None:
        raise ValueError("Tokenizer is required when decoder_type is 'greedy'")
        
    # Ensure everything is on the same device
    model = model.to(device)
    number_encoder = number_encoder.to(device)
    if intermediate_network is not None:
        intermediate_network = intermediate_network.to(device)
    
    # Set training mode for each component individually
    if not args.freeze_model:
        for param in model.parameters():
            param.requires_grad = True
        # Set training mode without recursion
        if hasattr(model, 'training'):
            object.__setattr__(model, 'training', True)
    else:
        for param in model.parameters():
            param.requires_grad = False
        # Set eval mode without recursion
        if hasattr(model, 'training'):
            object.__setattr__(model, 'training', False)
    
    number_encoder.train()
    if intermediate_network is not None:
        intermediate_network.train()
    
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        try:
            input_ids = batch['input_ids'].to(device)
            scatter_tensor = batch['scatter_tensor'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            last_token_mask = batch['last_token_mask'].to(device)
            len_gen = torch.randint(0, len_gen_size+1, (1,), device=device).item()
            
            # Get regular embeddings (now with gradients)
            regular_embeddings = get_regular_embeddings(model, input_ids)
            
            fourier_embeddings = number_encoder(scatter_tensor, len_gen=len_gen)
            # Apply intermediate network to the combined embeddings
            if intermediate_network is not None:
                fourier_embeddings = intermediate_network(fourier_embeddings)
            
            combined_embeddings = regular_embeddings + fourier_embeddings
            
            # modified part by EJ: match dtype of inputs same as model dtype
            combined_embeddings = combined_embeddings.to(device=device, dtype=model.dtype)
            attention_mask = attention_mask.to(device=device, dtype=model.dtype)
            fourier_embeddings = fourier_embeddings.to(device=device, dtype=model.dtype)
            
            
            
            if decoder_type == 'fne':
                # Forward pass through the model (now with gradients)
                if adapter_type is None:
                    outputs = model(inputs_embeds=combined_embeddings, attention_mask=attention_mask, output_hidden_states=True)
                elif adapter_type == 'linear' or adapter_type == 'affine' or adapter_type == 'low_rank':
                    outputs = model(inputs_embeds=combined_embeddings, attention_mask=attention_mask, output_hidden_states=True, fourier_embeddings=fourier_embeddings)
                else:
                    raise ValueError(f"Unsupported adapter type '{adapter_type}'.")
                before_decoder = outputs.hidden_states[-1]
                last_token_hidden_state = (before_decoder * last_token_mask.unsqueeze(-1)).sum(dim=1)
                loss = number_encoder.fourier_compute_loss(last_token_hidden_state, labels, int_digit_len, frac_digit_len, len_gen=len_gen)
            elif decoder_type == 'greedy':
                # Forward pass through the model (now with gradients)
                if adapter_type is None:
                    # Convert numeric labels to tokenized labels for greedy decoder
                    tokenized_labels = []
                    for label in labels:
                        # Convert numeric label to string
                        label_str = str(label.item())
                        # Add space prefix to ensure consistent tokenization
                        label_str = " " + label_str  # Add leading space for better tokenization
                        # Tokenize the string label
                        tokens = tokenizer(label_str, return_tensors="pt").input_ids.to(device)
                        # Handle special tokens carefully
                        if tokens.size(1) > 2:
                            tokens = tokens[:, 1:-1]  # Remove both BOS and EOS if present
                        elif tokens.size(1) > 1:
                            tokens = tokens[:, 1:]    # Remove just BOS if that's all we have
                        tokenized_labels.append(tokens.squeeze(0))
                    
                    # Pad tokenized labels to same length
                    max_len = max(t.size(0) for t in tokenized_labels)
                    padded_label_tokens = []
                    
                    # Check if any tokenization produced empty sequences
                    if max_len == 0:
                        logging.warning("Tokenization resulted in empty sequences. Check tokenizer settings.")
                        max_len = 1  # Set minimum length to avoid errors
                    
                    for tokens in tokenized_labels:
                        if tokens.size(0) < max_len:
                            padding = torch.full((max_len - tokens.size(0),), -100,  # Use -100 to ignore in loss
                                               dtype=tokens.dtype, device=tokens.device)
                            padded = torch.cat([tokens, padding])
                        else:
                            padded = tokens
                        padded_label_tokens.append(padded)
                    
                    # Stack to create batch
                    token_labels = torch.stack(padded_label_tokens)
                    
                    # Debug: Print token labels to check what we're feeding to the model
                    if args.debug:
                        for i in range(min(3, len(labels))):  # Print first few examples
                            original = str(labels[i].item())
                            tokenized = [t.item() for t in tokenized_labels[i]]
                            decoded = tokenizer.decode(tokenized_labels[i])
                            logging.debug(f"Label: {original}, Tokens: {tokenized}, Decoded: {decoded}")
                    
                    # Instead of directly using token_labels, prepare properly shifted labels for causal LM
                    # For causal LM models, we need to shift the labels right (this is typically done internally)
                    # But here we need to be explicit to ensure proper alignment
                    
                    # Create label mask where 1s mark positions to predict (end of sequence)
                    # We'll use this to create causal LM labels where only target positions have actual values
                    batch_size = input_ids.size(0)
                    seq_len = input_ids.size(1)
                    
                    # Create special labels tensor that only has the numeric tokens to predict
                    # First, create a tensor filled with -100 (ignored in loss computation)
                    shifted_labels = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
                    
                    # Place the actual label tokens at the end of the sequence based on the last token position
                    for i in range(batch_size):
                        # Find the position of the last non-padding token in the input sequence
                        last_pos = (input_ids[i] != tokenizer.pad_token_id).nonzero()[-1].item()
                        
                        # Calculate where to put label tokens (right after the last token) leaving room for labels
                        label_start_pos = last_pos - token_labels[i].size(0) + 1
                        
                        # Ensure we don't go out of bounds
                        if label_start_pos < 0:
                            label_start_pos = 0
                            logging.warning(f"Sequence too short for label tokens in batch {i}.")
                        
                        # Place the tokens at the right position
                        label_length = token_labels[i].size(0)
                        shifted_labels[i, label_start_pos:label_start_pos+label_length] = token_labels[i]
                        
                        # Debug
                        if args.debug and i < 3:
                            logging.debug(f"Input: {tokenizer.decode(input_ids[i])}")
                            logging.debug(f"Label position: {label_start_pos}, Label: {tokenizer.decode(token_labels[i][token_labels[i] != -100])}")
                            logging.debug(f"Shifted labels: {shifted_labels[i][shifted_labels[i] != -100]}")
                    
                    # Run model with properly aligned labels
                    outputs = model(
                        inputs_embeds=combined_embeddings, 
                        attention_mask=attention_mask,
                        labels=shifted_labels,  # Use the properly aligned labels
                        output_hidden_states=True
                    )
                    
                    # If loss is still too small, it might indicate a problem
                    if outputs.loss is not None and outputs.loss.item() < 1e-8:
                        logging.warning(f"Loss is extremely small: {outputs.loss.item()}. Using manual computation.")
                        
                        # Get logits for manual loss calculation
                        logits = outputs.logits
                        
                        # Calculate manual loss focusing on positions with labels
                        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                        
                        # Reshape for loss calculation (batch_size * sequence_length, vocab_size)
                        logits_reshaped = logits.view(-1, logits.size(-1))
                        labels_reshaped = shifted_labels.view(-1)
                        
                        # Calculate loss manually
                        manual_loss = loss_fct(logits_reshaped, labels_reshaped)
                        
                        # Use manual loss if it's better than the model's computation
                        if manual_loss.item() > outputs.loss.item():
                            outputs.loss = manual_loss
                            logging.warning(f"Using manual loss: {manual_loss.item()}")
                
                elif adapter_type == 'linear' or adapter_type == 'affine' or adapter_type == 'low_rank':
                    # Same tokenization process for adapter models
                    tokenized_labels = []
                    for label in labels:
                        label_str = str(label.item())
                        label_str = " " + label_str  # Add leading space for better tokenization
                        tokens = tokenizer(label_str, return_tensors="pt").input_ids.to(device)
                        # Handle special tokens carefully
                        if tokens.size(1) > 2:
                            tokens = tokens[:, 1:-1]  # Remove both BOS and EOS if present
                        elif tokens.size(1) > 1:
                            tokens = tokens[:, 1:]    # Remove just BOS if that's all we have
                        tokenized_labels.append(tokens.squeeze(0))
                    
                    # Pad tokenized labels to same length
                    max_len = max(t.size(0) for t in tokenized_labels) if tokenized_labels else 1
                    padded_label_tokens = []
                    for tokens in tokenized_labels:
                        if tokens.size(0) < max_len:
                            padding = torch.full((max_len - tokens.size(0),), -100,
                                               dtype=tokens.dtype, device=tokens.device)
                            padded = torch.cat([tokens, padding])
                        else:
                            padded = tokens
                        padded_label_tokens.append(padded)
                    
                    token_labels = torch.stack(padded_label_tokens)
                    
                    # Debug: Print token labels for adapter types as well
                    if args.debug:
                        for i in range(min(5, len(labels))):
                            original = str(labels[i].item())
                            tokenized = [t.item() for t in tokenized_labels[i]]
                            decoded = tokenizer.decode(tokenized_labels[i])
                            logging.debug(f"Adapter label: {original}, Tokens: {tokenized}, Decoded: {decoded}")
                    
                    # Create properly shifted labels for adapter models too
                    batch_size = input_ids.size(0)
                    seq_len = input_ids.size(1)
                    shifted_labels = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
                    
                    for i in range(batch_size):
                        last_pos = (input_ids[i] != tokenizer.pad_token_id).nonzero()[-1].item()
                        label_start_pos = last_pos - token_labels[i].size(0) + 1
                        
                        if label_start_pos < 0:
                            label_start_pos = 0
                            logging.warning(f"Sequence too short for label tokens in adapter batch {i}.")
                        
                        label_length = token_labels[i].size(0)
                        shifted_labels[i, label_start_pos:label_start_pos+label_length] = token_labels[i]
                    
                    outputs = model(
                        inputs_embeds=combined_embeddings, 
                        attention_mask=attention_mask,
                        fourier_embeddings=fourier_embeddings,
                        labels=shifted_labels,
                        output_hidden_states=True
                    )
                    
                    # Same manual loss fallback for adapter types
                    if outputs.loss is not None and outputs.loss.item() < 1e-8:
                        logging.warning(f"Adapter loss is extremely small: {outputs.loss.item()}")
                        
                        logits = outputs.logits
                        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                        logits_reshaped = logits.view(-1, logits.size(-1))
                        labels_reshaped = shifted_labels.view(-1)
                        manual_loss = loss_fct(logits_reshaped, labels_reshaped)
                        
                        if manual_loss.item() > outputs.loss.item():
                            outputs.loss = manual_loss
                            logging.warning(f"Using manual adapter loss: {manual_loss.item()}")
                
                else:
                    raise ValueError(f"Unsupported adapter type '{adapter_type}'.")
                loss = outputs.loss  # for regular training
                
                # Add extra loss check
                if loss is None or loss.item() == 0:
                    logging.warning("Loss is zero or None. This suggests a problem with tokenization, "
                                  "label alignment, or model configuration.")
                    
                    # Generate a dummy non-zero loss if all else fails
                    if loss is None or loss.item() == 0:
                        # Create a dummy loss from the combined embeddings 
                        # This is just to prevent zero loss and allow training to continue
                        dummy_loss = torch.mean(torch.abs(fourier_embeddings)) * 0.001
                        logging.warning(f"Creating non-zero dummy loss: {dummy_loss.item()}")
                        loss = dummy_loss
            else:
                raise ValueError(f"Unsupported decoder type '{decoder_type}'.")

            loss.backward()
            
            if args.clip:
                # Clip gradients for all parameters
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(number_encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(intermediate_network.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            
        except Exception as e:
            logging.error(f"Error processing batch {batch_idx}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            continue

    logging.info(f"avg Loss: {total_loss / len(train_loader)}")
    return total_loss / len(train_loader)

def train_regular(model, dataloader, optimizer, scheduler, device, args, number_encoder=None, intermediate_network=None):
    """
    Regular training loop for models without additional embedding modules.
    """
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()

        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    logging.info(f"avg Loss: {total_loss / len(dataloader)}")
    return total_loss / len(dataloader)

def train_xval(model, train_loader, xval, optimizer, scheduler, args, device):
    """
    Training loop for models using the xval module.
    """
    model.train()
    xval.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
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

        loss = xval.compute_loss(last_token_hidden_state, labels)

        loss.backward()
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(xval.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    logging.info(f"avg Loss: {total_loss / len(train_loader)}")
    return total_loss / len(train_loader)

def train_vanilla(model, train_loader, vanilla_model, intermediate_network, optimizer, scheduler, args, device):
    """
    Training loop for models using a vanilla embedding module with intermediate network.
    """
    model.train()
    vanilla_model.train()
    intermediate_network.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        scatter_tensor = batch['scatter_tensor'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        last_token_mask = batch['last_token_mask'].to(device)
        
        regular_embeddings = get_regular_embeddings(model, input_ids)
        vanilla_embeddings = vanilla_model(scatter_tensor)
        
        # Apply intermediate network to the combined embeddings
        combined_embeddings = regular_embeddings + vanilla_embeddings
        processed_embeddings = intermediate_network(combined_embeddings)

        outputs = model(inputs_embeds=processed_embeddings, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        last_token_hidden_state = (last_hidden_state * last_token_mask.unsqueeze(-1)).sum(dim=1)

        loss = vanilla_model.compute_loss(last_token_hidden_state, labels)
        
        loss.backward()
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(vanilla_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(intermediate_network.parameters(), max_norm=1.0)
            
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    logging.info(f"Training Loss: {avg_loss}")
    return avg_loss
