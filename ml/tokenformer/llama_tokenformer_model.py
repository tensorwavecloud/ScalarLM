
import logging
import time
import torch
from tokenformer.llama_tokenformer_layers import LlamaTokenformerDecoderLayer
from tokenformer.transformers_tokenformer import TransformersTokenformerSurgeon

def replace_layers(model, custom_layer_class, logger=logging.getLogger(__name__)):
    start_time = time.time()
    num_layers = len(model.model.layers)
    logger.info(f"(replace_layers inplace) Starting in-place layer modification for {num_layers} layers")
    
    for i in range(num_layers):
        layer_start = time.time()
        
        old_layer = model.model.layers[i]
        
        # Method 1: Try class replacement (if LlamaTokenformerDecoderLayer is just adding methods)
        try:
            # Store original class for fallback
            original_class = old_layer.__class__
            
            # Change the class of the existing instance
            old_layer.__class__ = custom_layer_class
            
            # If the custom layer needs initialization, call it
            if hasattr(custom_layer_class, '_initialize_tokenformer_components'):
                custom_layer_class._initialize_tokenformer_components(old_layer, model.config, i)
            
            layer_time = time.time() - layer_start
            
        except Exception as e:
            # Fallback to original method if in-place doesn't work
            logger.warning(f"(replace_layers inplace) In-place modification failed for layer {i}: {e}")
            logger.warning("(replace_layers inplace) Falling back to standard replacement")
            
            old_layer.__class__ = original_class  # Restore original class
            new_layer = custom_layer_class(model.config, i)
            new_layer.load_state_dict(old_layer.state_dict(), strict=False)
            model.model.layers[i] = new_layer
            layer_time = time.time() - layer_start
        
        if i < 5 or i % 20 == 0 or i == num_layers - 1:
            logger.info(f"Layer {i} in-place: {layer_time:.3f}s")
    
    total_time = time.time() - start_time
    logger.info(f"(replace_layers inplace) In-place layer modification completed: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    
    return model

def log_param_gradients(model, logger=logging.getLogger(__name__)):
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    total_count = sum(1 for p in model.parameters())
    logger.info(f"Parameter summary: {trainable_count:,} trainable out of {total_count:,} total")


def create_llama_tokenformer_model(model, device, train_lm_head=None):
    logger = logging.getLogger(__name__)
    overall_start = time.time()
    
    # Step 1: Replace layers
    logger.info("Starting layer replacement...")
    step1_start = time.time()
    model = replace_layers(model, LlamaTokenformerDecoderLayer, logger)
    step1_time = time.time() - step1_start
    logger.info(f"Layer replacement completed: {step1_time:.2f}s ({step1_time/60:.1f} minutes)")
    
    # Step 2: Insert adapter modules
    logger.info("Starting adapter module insertion...")
    step2_start = time.time()
    tokenformer_model = TransformersTokenformerSurgeon(model, device).insert_adapter_modules()

    step2_time = time.time() - step2_start
    logger.info(f"Adapter module insertion completed: {step2_time:.2f}s ({step2_time/60:.1f} minutes)")
    
    # Step 3: Count parameters for train_lm_head decision
    if train_lm_head is None:
        # Big models with more than 100M parameters don't need to train the lm_head
        # and getting the gradient scale right can be tricky.
        # Finally, the lm_head can be big and slow down adaptor loading in inference.
        logger.info("Counting parameters...")
        step3_start = time.time()
        param_count = count_parameters(tokenformer_model)
        step3_time = time.time() - step3_start
        train_lm_head = param_count < 100_000_000
        logger.info(f"Parameter counting completed: {step3_time:.2f}s, count={param_count:,}, train_lm_head={train_lm_head}")
    
    # Step 4: Freeze all parameters
    logger.info("Freezing all parameters...")
    step4_start = time.time()
    frozen_count = 0
    for param in tokenformer_model.parameters():
        param.requires_grad = False
        frozen_count += 1
    step4_time = time.time() - step4_start
    logger.info(f"Parameter freezing completed: {step4_time:.2f}s, frozen {frozen_count:,} parameters")
    
    # Step 5: Unfreeze tokenformer parameters
    logger.info("Unfreezing tokenformer parameters...")
    step5_start = time.time()
    unfrozen_count = 0
    for name, param in tokenformer_model.named_parameters():
        if any(module_name in name for module_name in ["tokenformer"]):
            param.requires_grad = True
            unfrozen_count += 1
    step5_time = time.time() - step5_start
    logger.info(f"Tokenformer parameter unfreezing completed: {step5_time:.2f}s, unfrozen {unfrozen_count:,} parameters")
    
    # Step 6: Handle lm_head training
    # If lm_head should be included in training, set it as well.
    # In some models, lm_head is tied to embeddings and not included as a param.
    # So it's best to access it directly.
    step6_start = time.time()
    if train_lm_head:
        logger.info("Setting lm_head for training...")
        if hasattr(tokenformer_model, 'lm_head') and hasattr(tokenformer_model.lm_head, 'weight'):
            tokenformer_model.lm_head.weight.requires_grad = True
            logger.info("lm_head weight set to trainable")
        else:
            logger.warning("lm_head or lm_head.weight not found")
    step6_time = time.time() - step6_start
    logger.info(f"lm_head handling completed: {step6_time:.2f}s")
    
    # Step 7: Log parameter gradients (optional, can be expensive)
    logger.info("Logging parameter gradients...")
    step7_start = time.time()
    log_param_gradients(tokenformer_model, logger)
    step7_time = time.time() - step7_start
    logger.info(f"Parameter gradient logging completed: {step7_time:.2f}s")
    
    total_time = time.time() - overall_start
    logger.info(f"create_llama_tokenformer_model total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    logger.info(f"Breakdown: layer_replace={step1_time:.1f}s, adapter_insert={step2_time:.1f}s, param_count={step3_time:.1f}s, freeze={step4_time:.1f}s, unfreeze={step5_time:.1f}s, lm_head={step6_time:.1f}s, logging={step7_time:.1f}s")
    
    return tokenformer_model

# Define a function to count parameters
def count_parameters(module):
    return sum(p.numel() for p in module.parameters())

