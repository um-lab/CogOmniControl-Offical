import torch


def _subseq_index(seq, sub):
    n, m = len(seq), len(sub)
    if m == 0 or n < m:
        return -1
    for i in range(n - m + 1):
        if seq[i:i + m] == sub:
            return i
    return -1


def _find_marker(prompt_ids, tokenizer, marker_text,
                 fallbacks=("（评估器库）", "Evaluator Registry", "评估器库")):
    marker_ids = tokenizer.encode(marker_text, add_special_tokens=False)
    idx = _subseq_index(prompt_ids, marker_ids)
    if idx != -1:
        return idx, len(marker_ids)
    for fb in fallbacks:
        fb_ids = tokenizer.encode(fb, add_special_tokens=False)
        idx = _subseq_index(prompt_ids, fb_ids)
        if idx != -1:
            return idx, len(fb_ids)
    return -1, 0


def _build_single_layer_embeds(outputs, layer_idx, cut_idx, marker_len, keep_marker):
    """
    Args:
        outputs: outputs.hidden_states 
        layer_idx (int)
        cut_idx (int): 
        marker_len (int):
        keep_marker (bool): 

    Returns:
        Tensor:  (b, total_len, hidden_size)，total_len =  prompt_len + answer token
    """
    prompt_hs = outputs.hidden_states[0][layer_idx]  # (b, prompt_len, hidden_size)
    if cut_idx > 0:
        end = cut_idx + marker_len if keep_marker else cut_idx
        end = min(end, prompt_hs.shape[1])
        prompt_hs = prompt_hs[:, :end, :]
    answer_hs = [
        step_hidden_states[layer_idx]
        for step_hidden_states in outputs.hidden_states[1:]
    ]  # List[(b, 1, hidden_size)]

    layer_embeds = torch.cat([prompt_hs] + answer_hs, dim=1)  # (b, total_len, hidden_size)
    return layer_embeds


def build_llm_embeds(outputs, input_ids, tokenizer,
                     marker_text="# Evaluator Registry（评估器库）",
                     keep_marker=False,
                     layer_indices=None):

    prompt_ids = input_ids[0].tolist()
    cut_idx, marker_len = _find_marker(prompt_ids, tokenizer, marker_text)

    # 未指定 layer_indices 时，保持原始行为：只取最后一层。
    if layer_indices is None:
        layer_indices = [-1]

    # 逐层构建各自完整的 (prompt + answer) 序列，暂存到列表中。
    per_layer_embeds = [
        _build_single_layer_embeds(outputs, layer_idx, cut_idx, marker_len, keep_marker)
        for layer_idx in layer_indices
    ]  # List[(b, total_len, hidden_size)]，长度 = len(layer_indices)

    # 将不同层的完整序列，继续沿序列长度维（dim=1）依次拼接在一起，
    # 特征维（hidden_size，最后一维）保持不变。
    # 例如 layer_indices=[16, 24, 36] 时：
    #   llm_embeds = concat([layer16_full_seq, layer24_full_seq, layer36_full_seq], dim=1)
    #   -> (b, total_len*3, hidden_size)
    llm_embeds = torch.cat(per_layer_embeds, dim=1)

    return llm_embeds, cut_idx
