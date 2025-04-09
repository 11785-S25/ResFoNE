import torch
import logging
from train.utils import get_regular_embeddings, handle_nan_loss

def train_fne(model, train_loader, number_encoder, intermediate_network, optimizer, scheduler, args, int_digit_len, frac_digit_len, len_gen_size, decoder_type, adapter_type, device):
    """
    Training loop for Fourier Neural Embedding (FNE) based models with intermediate network.
    LLM parameters are now trainable along with FNE and intermediate network.
    """
    # Set all models to training mode
    if not args.freeze_model:
        model.train()
    else:
        model.eval()
    number_encoder.train()
    if intermediate_network is not None:
        intermediate_network.train()
    
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
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
        combined_embeddings = combined_embeddings.to(model.dtype)
        attention_mask = attention_mask.to(model.dtype)

        # Forward pass through the model (now with gradients)
        if adapter_type is None:
            outputs = model(inputs_embeds=combined_embeddings, attention_mask=attention_mask, output_hidden_states=True)
        elif adapter_type == 'linear' or adapter_type == 'low_rank':
            outputs = model(inputs_embeds=combined_embeddings, attention_mask=attention_mask, output_hidden_states=True)
        else:
            raise ValueError(f"Unsupported adapter type '{adapter_type}'.") 
        

        if decoder_type == 'fourier':
            before_decoder = outputs.hidden_states[-1]
            last_token_hidden_state = (before_decoder * last_token_mask.unsqueeze(-1)).sum(dim=1)
            loss = number_encoder.fourier_compute_loss(last_token_hidden_state, labels, int_digit_len, frac_digit_len, len_gen=len_gen)
        elif decoder_type == 'greedy':
            loss = outputs.loss
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

    logging.info(f"avg Loss: {total_loss / len(train_loader)}")
    return total_loss / len(train_loader)

def train_regular(model, dataloader, optimizer, scheduler, device, args):
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
