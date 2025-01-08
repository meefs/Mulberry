import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import MllamaForConditionalGeneration



def init_model(args):

    activated_models = []
    model_dict={}
    if args.gpt_version is not None:
        activated_models.append(args.gpt_version)

    # qwen2vl_7b
    if args.qwen2_vl_7b_model_path is not None:
        print('init qwen2 vl 7b model')
        qwen2_vl_7b_model  = Qwen2VLForConditionalGeneration.from_pretrained(
            args.qwen2_vl_7b_model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation='flash_attention_2',
        )
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        qwen2_vl_7b_processor = AutoProcessor.from_pretrained(args.qwen2_vl_7b_model_path, min_pixels=min_pixels, max_pixels=max_pixels)

        activated_models.append('qwen2_vl_7b')
        model_dict['qwen2_vl_7b'] = {'model': qwen2_vl_7b_model, 'processor': qwen2_vl_7b_processor}

    # qwen2vl_2b
    if args.qwen2_vl_2b_model_path is not None:
        print('init qwen2 vl 2b model')
        qwen2_vl_2b_model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.qwen2_vl_2b_model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation='flash_attention_2',
        )
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        qwen2_vl_2b_processor = AutoProcessor.from_pretrained(args.qwen2_vl_2b_model_path, min_pixels=min_pixels, max_pixels=max_pixels)

        activated_models.append('qwen2_vl_2b')
        model_dict['qwen2_vl_2b'] = {'model': qwen2_vl_2b_model, 'processor': qwen2_vl_2b_processor}

    # qwen2vl_72b
    if args.qwen2_vl_72b_model_path is not None:
        print('init qwen2 vl 72b model')
        qwen2_vl_72b_model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.qwen2_vl_72b_model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation='flash_attention_2',
        )
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        qwen2_vl_72b_processor = AutoProcessor.from_pretrained(args.qwen2_vl_72b_model_path, min_pixels=min_pixels, max_pixels=max_pixels)

        activated_models.append('qwen2_vl_72b')
        model_dict['qwen2_vl_72b'] = {'model': qwen2_vl_72b_model, 'processor': qwen2_vl_72b_processor}

    # llama3.2_vision_11b
    if args.llama3_vision_11b_model_path is not None:
        print('init llama3.2 vision 11b model')
        llama_vision_11b_model = MllamaForConditionalGeneration.from_pretrained(
            args.llama3_vision_11b_model_path, torch_dtype=torch.bfloat16, device_map="auto",
        )
        llama_vision_11b_processor = AutoProcessor.from_pretrained(args.llama3_vision_11b_model_path)

        activated_models.append('llama_vision_11b')
        model_dict['llama_vision_11b'] = {'model': llama_vision_11b_model, 'processor': llama_vision_11b_processor}

    if args.llava_next_8b_model_path is not None:
        llava_next_8b_model = LlavaNextForConditionalGeneration.from_pretrained(
            args.llava_next_8b_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation='flash_attention_2'
            # use_flash_attention_2=True,
        )
        llava_next_8b_processor = LlavaNextProcessor.from_pretrained(args.llava_next_8b_model_path)

        activated_models.append('llava_next_8b')
        model_dict['llava_next_8b'] = {'model': llava_next_8b_model, 'processor': llava_next_8b_processor}
        

    return activated_models, model_dict

