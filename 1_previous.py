import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch import tensor
import pickle, numpy as np, copy
def savelog(title, variable):
    """
    Appends a log entry to 'log.txt' with the given title and variable.
    Each entry consists of the title on one line, the variable on the next line,
    followed by three newline characters to separate entries.
    
    :param title: The title of the log entry (string).
    :param variable: The variable to log (will be converted to string).
    """
    with open("log_copy.txt", "a", encoding="utf-8") as file:
        file.write(title + "\n")
        file.write(str(variable) + "\n\n\n")

def judge_two_arrays(a, b):
    diff = np.abs(a - b)
    max_diff = np.max(diff)
    p80 = np.percentile(diff, 80)
    return max_diff, p80


def clear_log_file():
    """
    Clears all contents of the 'log.txt' file by truncating it.
    If the file does not exist, it will be created as an empty file.
    """
    with open("log.txt", "w", encoding="utf-8") as file:
        pass  # Opening in 'w' mode truncates the file to zero length
    with open("log_copy.txt", "w", encoding="utf-8") as file:
        pass  # Opening in 'w' mode truncates the file to zero length
clear_log_file()

# Prepare the input as before
chat = [
    {"role": "user", "content": "who are you? please speak chinese to me"},
]

# 1: Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("/WeijieInternship/DS_R1_Qwen_1.5B", device_map="auto", torch_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(r"/WeijieInternship/DS_R1_Qwen_1.5B", device_map="auto", torch_dtype=torch.bfloat16,low_cpu_mem_usage=True)
# model = AutoModelForCausalLM.from_pretrained("/root/numpy_ml/DeepSeek-R1-Distill-Qwen-1.5B")
# print(model)

# 2: Apply the chat template
formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print("Formatted chat:\n", formatted_chat)

# 3: Tokenize the chat (This can be combined with the previous step using tokenize=True)
inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
# Move the tokenized inputs to the same device the model is on (GPU/CPU)
# inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
# input_ids = inputs['input_ids'].numpy()
print("Tokenized inputs:\n", inputs)

input_ids = inputs['input_ids'].numpy()
print("\ninput_id shape: \n",input_ids.shape, end='\n'*5)

outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.1)
print("Generated tokens:\n", outputs)

#开始复现

import safetensors
all_parameters = safetensors.torch.load_file("/WeijieInternship/DS_R1_Qwen_1.5B/model.safetensors")
for key in all_parameters.keys():
    print(key, all_parameters[key].shape, sep='\t'*10)

# transforms库版本不同，可能对应的代码行数不完全一致
# embedding：对应位置在transformers库 /usr/local/lib/python3.8/dist-packages/transformers/models/qwen2/modeling_qwen2.py
# 853行代码  if inputs_embeds is None:
#              inputs_embeds = self.embed_tokens(input_ids)
embed_tokens_weight = all_parameters['model.embed_tokens.weight'].float().numpy()
hidden_states = embed_tokens_weight[input_ids]
#print(hidden_states.shape, '\n'*5)

#复现rotary embedding
position_ids = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]], dtype=np.float32)
batch_size = position_ids.shape[0]
position_ids_expanded = position_ids[:, None, :]
inv_freq = model.model.rotary_emb.inv_freq.cpu().float().numpy().astype(np.float32)
inv_freq_expanded = inv_freq[None, :, None].astype(np.float32)
inv_freq_expanded = np.broadcast_to(inv_freq_expanded, (batch_size, inv_freq.shape[0], 1))

freqs = inv_freq_expanded * position_ids_expanded
freqs = freqs.transpose(0, 2, 1)
emb = np.concatenate((freqs, freqs), axis=-1)
cos = np.cos(emb).astype(np.float32)
sin = np.sin(emb).astype(np.float32)
attention_scaling=1
cos = cos * attention_scaling
sin = sin * attention_scaling




#decoder layer

# Define the linear function
def linear(input, weight, bias):
    output = np.dot(input, weight.T) + bias
    return output

#复现apply rotary pos emb
def rotate_half(x):
    """
    Rotates half the hidden dimensions of the input.
    
    Args:
        x (np.ndarray): Input array of shape [..., head_dim].
    
    Returns:
        np.ndarray: Rotated array with the same shape.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]          # First half of head_dim
    x2 = x[..., half:]          # Second half of head_dim
    return np.concatenate((-x2, x1), axis=-1)  # Concatenate along last dimension

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to the query and key tensors using NumPy.
    
    Args:
        q (np.ndarray): Query tensor, shape [batch_size, num_heads, seq_len, head_dim].
        k (np.ndarray): Key tensor, shape [batch_size, num_heads, seq_len, head_dim].
        cos (np.ndarray): Cosine tensor, shape [batch_size, seq_len, head_dim].
        sin (np.ndarray): Sine tensor, shape [batch_size, seq_len, head_dim].
        unsqueeze_dim (int): Dimension to expand cos and sin (default=1).
    
    Returns:
        tuple: (q_embed, k_embed), each with shape [batch_size, num_heads, seq_len, head_dim].
    """
    # Expand cos and sin to match q and k shapes
    cos = np.expand_dims(cos, axis=unsqueeze_dim)
    sin = np.expand_dims(sin, axis=unsqueeze_dim)
    
    # Apply RoPE transformation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep=6):
    """
    NumPy implementation of PyTorch's repeat_kv function. Repeats key/value heads to match attention heads.
    
    Args:
        hidden_states (np.ndarray): Input array of shape (batch, num_key_value_heads, slen, head_dim).
        n_rep (int): Number of times to repeat each key/value head.
    
    Returns:
        np.ndarray: Output array of shape (batch, num_key_value_heads * n_rep, slen, head_dim).
    """
    # Extract input dimensions
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    
    # If n_rep is 1, return the input unchanged
    if n_rep == 1:
        return hidden_states
    
    # Add a new axis and repeat along it
    hidden_states_expanded = np.expand_dims(hidden_states, axis=2)  # Shape: (batch, num_key_value_heads, 1, slen, head_dim)
    hidden_states_repeated = np.repeat(hidden_states_expanded, n_rep, axis=2)  # Shape: (batch, num_key_value_heads, n_rep, slen, head_dim)
    
    # Reshape to combine num_key_value_heads and n_rep into num_attention_heads
    output = hidden_states_repeated.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    return output

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, 
                                 is_causal=False, scale=None, enable_gqa=False):
    """
    NumPy implementation of scaled dot-product attention.
    
    Args:
        query (np.ndarray): Shape [batch_size, num_heads, L, head_dim]
        key (np.ndarray): Shape [batch_size, num_heads, S, head_dim]
        value (np.ndarray): Shape [batch_size, num_heads, S, head_dim]
        attn_mask (np.ndarray): Shape [L, S], optional attention mask
        dropout_p (float): Dropout probability
        is_causal (bool): Whether to apply causal masking
        scale (float): Scaling factor
        enable_gqa (bool): Whether to enable Grouped Query Attention
    
    Returns:
        np.ndarray: Attention output, shape [batch_size, num_heads, L, head_dim]
    """
    # Get sequence lengths
    L, S = query.shape[-2], key.shape[-2]
    
    # Initialize attention bias
    attn_bias = np.zeros((L, S), dtype=query.dtype)
    
    # Apply causal masking if enabled
    if is_causal:
        # Create a lower triangular mask (1s on and below diagonal, 0s above)
        temp_mask = np.ones((L, S), dtype=bool)
        temp_mask = np.tril(temp_mask, k=0)  # Lower triangle including diagonal
        # Set upper triangle to -inf in attn_bias
        attn_bias[~temp_mask] = float('-inf')
    
    # Apply provided attn_mask (assuming it's already in correct format)
    if attn_mask is not None:
        attn_bias = attn_mask + attn_bias  # Directly add, as per your instruction
    
    # Handle Grouped Query Attention (GQA) if enabled
    if enable_gqa:
        num_repeats = query.shape[-3] // key.shape[-3]
        key = np.repeat(key, num_repeats, axis=-3)
        value = np.repeat(value, num_repeats, axis=-3)
    
    # Compute attention weights
    # Transpose key: [batch_size, num_heads, S, head_dim] -> [batch_size, num_heads, head_dim, S]
    key_T = np.transpose(key, (0, 1, 3, 2))
    # Matrix multiplication: query @ key^T
    attn_weight = np.matmul(query, key_T) * scale
    # Add attention bias
    attn_weight += attn_bias[np.newaxis, np.newaxis, :, :]  # Broadcast to [batch_size, num_heads, L, S]
    
    # Softmax along the last dimension (S)
    attn_weight = np.exp(attn_weight - np.max(attn_weight, axis=-1, keepdims=True))  # Numerical stability
    attn_weight = attn_weight / np.sum(attn_weight, axis=-1, keepdims=True)
    
    # Apply dropout
    if dropout_p > 0:
        mask = np.random.random(attn_weight.shape) < (1 - dropout_p)
        attn_weight = attn_weight * mask / (1 - dropout_p)  # Scale to maintain expected value
    
    # Compute final output
    output = np.matmul(attn_weight, value)
    
    return output



def silu(x):
    """SiLU (Swish) activation: x * sigmoid(x)"""
    return x / (1 + np.exp(-x))









for decoder_layer in range (28):
    # decoder_layer=0
    print("my decoder layer:",decoder_layer)
    residual=hidden_states
    # layer norm：对应位置在transformers库 /usr/local/lib/python3.8/dist-packages/transformers/models/qwen2/modeling_qwen2.py
    # 620行代码  hidden_states = self.input_layernorm(hidden_states)
    hidden_states = hidden_states.astype(np.float32)
    rms = np.mean(np.square(hidden_states), axis=-1, keepdims=True)
    hidden_states = hidden_states * 1 / np.sqrt(rms + 1e-6)
    layers_input_layernorm_weight = all_parameters[f'model.layers.{decoder_layer}.input_layernorm.weight'].float().numpy()
    layers_input_layernorm_weight = layers_input_layernorm_weight.astype(np.float32)
    hidden_states = layers_input_layernorm_weight * hidden_states
    hidden_states = hidden_states.astype(np.float16)
    # print(hidden_states.shape, '\n'*5)




    if decoder_layer==1 or decoder_layer==0:
        savelog("layer norm",hidden_states)

    #self attention: Qwen2DecoderLayer 里的 hidden_states, self_attn_weights = self.self_attn(...）
    #calling the following subroutine:
    #   apply_rotary_pos_emb(): embed position into the key-value
    #   eager_attention_forward(): perform the main calculations, and apply softmax&dropout layer
    #   repeat_kv(): The hidden states go from (batch,num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    # 



    o_weight = all_parameters[f'model.layers.{decoder_layer}.self_attn.o_proj.weight'].float().numpy()
    q_weight = all_parameters[f'model.layers.{decoder_layer}.self_attn.q_proj.weight'].float().numpy()
    q_bias = all_parameters[f'model.layers.{decoder_layer}.self_attn.q_proj.bias'].float().numpy()
    v_weight = all_parameters[f'model.layers.{decoder_layer}.self_attn.v_proj.weight'].float().numpy()
    v_bias = all_parameters[f'model.layers.{decoder_layer}.self_attn.v_proj.bias'].float().numpy()
    k_weight = all_parameters[f'model.layers.{decoder_layer}.self_attn.k_proj.weight'].float().numpy()
    k_bias = all_parameters[f'model.layers.{decoder_layer}.self_attn.k_proj.bias'].float().numpy()
    num_heads=-1 #32 from configuration_qwen2.py: num_attention_heads， -1 from debuging the hidden shape of the first token
    head_dim=4096//32 #hidden_size // num_heads

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, num_heads, head_dim)

    # Compute query, key, value states
    query_states = linear(hidden_states, q_weight, q_bias)
    key_states = linear(hidden_states, k_weight, k_bias)
    value_states = linear(hidden_states, v_weight, v_bias)

    # Reshape to [batch_size, seq_len, num_heads, head_dim]
    query_states = query_states.reshape(*hidden_shape)
    key_states = key_states.reshape(*hidden_shape)
    value_states = value_states.reshape(*hidden_shape)

    # Transpose to [batch_size, num_heads, seq_len, head_dim]
    query_states = np.transpose(query_states, (0, 2, 1, 3))
    key_states = np.transpose(key_states, (0, 2, 1, 3))
    value_states = np.transpose(value_states, (0, 2, 1, 3))




    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    # checked

    #scaled dot product attention (SDPA)




    key_states = repeat_kv(key_states)
    value_states = repeat_kv(value_states)
    # checked


    causal_mask=None
    dropout=0
    scaling=head_dim**-0.5
    is_causal=True
    output = scaled_dot_product_attention(query_states, key_states, value_states,attn_mask=causal_mask, dropout_p=dropout, scale=scaling, is_causal=is_causal) 



    # Transpose dimensions 1 and 2
    attn_output = np.transpose(output, (0, 2, 1, 3))
    # checked


    #scaled dot product attention (SDPA) done

    # Ensure contiguity (optional in NumPy, but included for fidelity to PyTorch)
    attn_output = np.ascontiguousarray(attn_output)
    attn_output = attn_output.reshape(*input_shape, -1)
    attn_output = linear(attn_output, o_weight, 0)
    attn_weights=None
    # checked


    savelog("attention output:",attn_output)
    #attention done. back to decoder
    hidden_states=attn_output
    self_attn_weights=attn_weights





    hidden_states=residual+hidden_states
    residual=hidden_states
    if decoder_layer==1 or decoder_layer==0:
        savelog("self attn",hidden_states)



    #post_attention_layernorm
    hidden_states = hidden_states.astype(np.float32)
    rms = np.mean(np.square(hidden_states), axis=-1, keepdims=True)
    hidden_states = hidden_states * 1 / np.sqrt(rms + 1e-6)
    layers_input_layernorm_weight = all_parameters[f'model.layers.{decoder_layer}.post_attention_layernorm.weight'].float().numpy()
    layers_input_layernorm_weight = layers_input_layernorm_weight.astype(np.float32)
    hidden_states = layers_input_layernorm_weight * hidden_states
    hidden_states = hidden_states.astype(np.float16)

    if decoder_layer==1 or decoder_layer==0:
        savelog("post attn norm",hidden_states)



    # mlp



    down_weight = all_parameters[f'model.layers.{decoder_layer}.mlp.down_proj.weight'].float().numpy()
    gate_weight = all_parameters[f'model.layers.{decoder_layer}.mlp.gate_proj.weight'].float().numpy()
    up_weight = all_parameters[f'model.layers.{decoder_layer}.mlp.up_proj.weight'].float().numpy()
    hidden_size=4096#from config
    intermediate_size=22016#from config
    gate_proj=linear(hidden_states,gate_weight,0)
    act=silu(gate_proj)
    up_proj=linear(hidden_states,up_weight,0)
    down_proj=linear(act*up_proj,down_weight,0)
    hidden_states=down_proj
    #mlp finished, back to decoder



    if decoder_layer==1 or decoder_layer==0:
        savelog("mlp",hidden_states)

    hidden_states=residual+hidden_states

    savelog("decoder output",hidden_states)

    #test for each decoder layer

    if decoder_layer==6:
        import pickle,time
        with open(r'./test.pkl', 'rb') as f:
            aim_hidden_states = pickle.load(f)
            aim_hidden_states = [tensor.cpu().float().numpy() for tensor in aim_hidden_states]
        print("saving")

        # 比较库代码保存的中间结果，和自己用numpy复现的中间结果的差
        results=judge_two_arrays(hidden_states, aim_hidden_states)
        print("max_diff, p80=",results)
        time.sleep(20)


    #decoder layer done, back to model



#test for 28 decoder layers

# import pickle,time
# with open(r'./test.pkl', 'rb') as f:
#     aim_hidden_states = pickle.load(f)
#     aim_hidden_states = [tensor.cpu().float().numpy() for tensor in aim_hidden_states]
# print("saving")

# # 比较库代码保存的中间结果，和自己用numpy复现的中间结果的差
# results=judge_two_arrays(hidden_states, aim_hidden_states)
# print("max_diff, p80=",results)
# time.sleep(20)


#norm
hidden_states = hidden_states.astype(np.float32)
rms = np.mean(np.square(hidden_states), axis=-1, keepdims=True)
hidden_states = hidden_states * 1 / np.sqrt(rms + 1e-6)
layers_input_layernorm_weight = all_parameters[f'model.layers.{decoder_layer}.input_layernorm.weight'].float().numpy()
layers_input_layernorm_weight = layers_input_layernorm_weight.astype(np.float32)
hidden_states = layers_input_layernorm_weight * hidden_states
hidden_states = hidden_states.astype(np.float16)

#model done


#test for after normalization

# import pickle,time
# with open(r'./test.pkl', 'rb') as f:
#     aim_hidden_states = pickle.load(f)
#     aim_hidden_states = [tensor.cpu().float().numpy() for tensor in aim_hidden_states]
# print("saving")

# # 比较库代码保存的中间结果，和自己用numpy复现的中间结果的差
# results=judge_two_arrays(hidden_states, aim_hidden_states)
# print("max_diff, p80=",results)
# time.sleep(20)

























# lm_weight = all_parameters['lm_head.weight'].float().numpy()
# vocab_size=151936
# num_logits_to_keep=0
# hidden_state_slice=hidden_states[:, -num_logits_to_keep:, :]
# logits=linear(hidden_state_slice, lm_weight, 0)
# savelog("logit",logits)







# # lwjedit 在库文件中加上
# import pickle
# with open(r'./test.pkl', 'wb') as f:
#     pickle.dump([hidden_states], f)
# import time
# print("saving")
# time.sleep(10)



# import pickle
# with open(r'./test.pkl', 'rb') as f:
#     aim_hidden_states = pickle.load(f)
#     aim_hidden_states = [tensor.cpu().float().numpy() for tensor in aim_hidden_states]
# import time
# print("saving")
# time.sleep(10)

# # 比较库代码保存的中间结果，和自己用numpy复现的中间结果的差
# results=judge_two_arrays(hidden_states, aim_hidden_states)
# print("max_diff, p80=",results)