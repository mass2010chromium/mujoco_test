"""
Abstractions for dealing with the Transformers AutoModel/AutoProcessor API
"""

def generate_output(model, processor, messages, prior_text='', prior_images=[], thinking_mode='auto', device='cuda', max_new_tokens=512, **kwargs):
    """
    Generate some output from a VLM with the AutoProcessor/AutoModel interface.

    Messages are the normal content/type/role messages.
    Prior text/images are the previous context, since R4B doesn't have a clean way
    of adding generated output back to the messages list.
    
    Return: (response, new_text_context, new_image_context, eos_flag, num_new_tokens)
    """
    import torch

    images = []
    for turn in messages:
        for message in turn['content']:
            if message['type'] == 'image':
                images.append(message['image'])

    # This isn't quite tokenization, it just puts tags like <|im_start|> for turns or <think> to begin thinking
    if len(messages) > 0:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            thinking_mode=thinking_mode
        )
        text_context = prior_text + text
    else:
        text_context = prior_text

    #Concatenate text and image contexts
    image_context = prior_images+images
    inputs = processor(
        images=image_context if len(image_context) else None,
        text=text_context,
        return_tensors='pt'
    ).to(device)

    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
    output_ids = generated_ids[0][len(inputs.input_ids[0]):]
    eos_id = processor.tokenizer.eos_token_id
    eos_output = torch.any(output_ids == eos_id)
    output_text = processor.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text, text_context, image_context, eos_output, len(generated_ids)


def make_message(texts=[], images=[], role="user"):
    """
    Makes a message structure containing the appropriate information for the given data.
    TODO: support video.
    """
    messages = []
    for image in images:
        messages.append({'type': 'image', 'image': image})
    if type(texts) == str:
        messages.append({'type': 'text', 'text': texts})
    else:
        for text in texts:
            messages.append({'type': 'text', 'text': text})
    return {"role": role, "content": messages}
