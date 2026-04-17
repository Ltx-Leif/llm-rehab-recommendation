# llm_rehab_recommendation-后端demo/huatuoGPT-Vision/cli.py

import sys
import os
from typing import Any, List  # 确保 List 被导入

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
print(dir_path)  # 打印当前脚本所在目录，用于确认路径
sys.path.insert(0, dir_path)  # 将脚本所在目录添加到 python 路径，以便导入 llava

from llava.constants import IMAGE_TOKEN_INDEX
# from llava.model import * # 作者的原始导入，如果 LlavaQwen2ForCausalLM 导入有问题，可能需要更具体的导入

from transformers import AutoTokenizer
from transformers import TextIteratorStreamer
from threading import Thread
import torch
import logging
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class HuatuoChatbot():
    def __init__(self, model_dir, device='cuda'):  # device 参数现在由命令行提供
        self.model_dir = model_dir
        self.abs_model_dir = os.path.abspath(model_dir)

        self.gen_kwargs = {
            'do_sample': True,
            'max_new_tokens': 512,
            'min_new_tokens': 1,
            'temperature': .2,
            'repetition_penalty': 1.2
            # eos_token_id 和 pad_token_id 会在 init_components 中设置
        }
        self.device = device  # 使用传入的 device
        logger.info(f"HuatuoChatbot will use device: {self.device}")

        self.init_components()
        self.history = []
        self.images = []  # 用于累积 chat 模式下的图像
        self.debug = True
        self.max_image_num = 6

    def init_components(self):
        logger.info(f"Initializing components from model_dir: {self.abs_model_dir}")
        d = self.abs_model_dir

        model = None
        loading_info = {}  # 初始化为字典
        tokenizer_local = None
        image_processor_local = None

        # --- 根据模型目录名或 config.json 选择加载逻辑 ---
        # 首先尝试基于字符串匹配，如果失败则尝试读取 config.json
        loaded_by_string_match = False
        if 'huatuogpt-vision' in d.lower():
            logger.info(f"Attempting to load LlavaQwen2ForCausalLM (primary) from {self.abs_model_dir}")
            try:
                from llava.model.language_model.llava_qwen2 import LlavaQwen2ForCausalLM
                model, loading_info = LlavaQwen2ForCausalLM.from_pretrained(
                    self.abs_model_dir,
                    init_vision_encoder_from_ckpt=True,
                    output_loading_info=True,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device  # 使用传入的 device
                )
                tokenizer_local = AutoTokenizer.from_pretrained(self.abs_model_dir)
                loaded_by_string_match = True
                logger.info("LlavaQwen2ForCausalLM loaded via string match.")
            except Exception as e:
                logger.warning(f"Failed to load LlavaQwen2ForCausalLM via string match: {e}. Will try config.json.")

        # 如果字符串匹配失败或第一个条件不满足，尝试基于 config.json
        if not loaded_by_string_match:
            config_path = os.path.join(self.abs_model_dir, "config.json")
            model_type_from_config = None
            if os.path.exists(config_path):
                try:
                    import json
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    model_type_from_config = config_data.get("model_type")
                    logger.info(f"Read model_type '{model_type_from_config}' from config.json")
                except Exception as e:
                    logger.warning(f"Could not read model_type from config.json: {e}")

            if model_type_from_config == "llava_qwen2":
                logger.info(f"Loading LlavaQwen2ForCausalLM based on config.json from {self.abs_model_dir}")
                from llava.model.language_model.llava_qwen2 import LlavaQwen2ForCausalLM
                model, loading_info = LlavaQwen2ForCausalLM.from_pretrained(
                    self.abs_model_dir,
                    init_vision_encoder_from_ckpt=True,
                    output_loading_info=True,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device
                )
                tokenizer_local = AutoTokenizer.from_pretrained(self.abs_model_dir)
            elif model_type_from_config == "llava_llama":  # 假设如果 config.json 中是 llava_llama
                logger.info(f"Loading LlavaLlamaForCausalLM based on config.json from {self.abs_model_dir}")
                from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
                model, loading_info = LlavaLlamaForCausalLM.from_pretrained(
                    self.abs_model_dir,
                    init_vision_encoder_from_ckpt=True,
                    output_loading_info=True,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device
                )
                tokenizer_local = AutoTokenizer.from_pretrained(self.abs_model_dir)
            else:
                logger.error(f"Model type not recognized by cli.py logic for model_dir: {d}. "
                             f"String match failed. Checked config: {model_type_from_config}")
                raise NotImplementedError("Model type not recognized or loading failed.")

        # --- 后续处理，确保 model, tokenizer_local, image_processor_local 被赋值 ---
        if model is None:
            # 防御性检查
            raise RuntimeError("Model was not loaded by any branch.")
        if tokenizer_local is None:
            tokenizer_local = AutoTokenizer.from_pretrained(self.abs_model_dir)  # 作为后备

        missing_keys = loading_info.get('missing_keys', [])
        unexpected_keys = loading_info.get('unexpected_keys', [])
        logger.info(f"Model loading info - Missing keys: {len(missing_keys)}, "
                    f"Unexpected keys: {len(unexpected_keys)}")
        if unexpected_keys:
            logger.debug(f"Unexpected keys detail: {unexpected_keys[:5]}...")

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            logger.info("Loading vision tower model...")
            vision_tower.load_model()
        logger.info(f"Vision tower device after potential loading: {vision_tower.device}")

        if hasattr(vision_tower, 'image_processor'):
            image_processor_local = vision_tower.image_processor
            logger.info(f"Image processor type: {type(image_processor_local)}")
            if image_processor_local is None:
                logger.error("vision_tower.image_processor IS NONE!")
        else:
            logger.error("vision_tower DOES NOT HAVE image_processor attribute!")
            # 尝试从 tokenizer 加载，作为后备（不常见，但有些模型可能这样）
            # 或者直接使用 vision_tower 本身作为 processor
            if isinstance(vision_tower, torch.nn.Module) and hasattr(vision_tower, "preprocess"):
                logger.warning("Using vision_tower itself as image_processor (experimental).")
                image_processor_local = vision_tower
            else:
                # 如果无法获取 image_processor，这是个严重问题
                raise RuntimeError("Could not obtain image_processor from vision_tower.")

        if image_processor_local is None:  # 再次检查
            raise RuntimeError("Failed to initialize image_processor_local.")

        model.eval()
        self.model = model  # device_map 应该已经处理了设备放置
        logger.info(f"Model final device: {self.model.device}")

        self.tokenizer = tokenizer_local
        self.processor = image_processor_local

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info(f"pad_token_id was None, set to eos_token_id: {self.tokenizer.eos_token_id}")

        self.gen_kwargs['eos_token_id'] = self.tokenizer.eos_token_id
        self.gen_kwargs['pad_token_id'] = self.tokenizer.pad_token_id

    def clear_history(self):
        self.images = []
        self.history = []
        logger.info("History and images cleared.")

    def tokenizer_image_token(self, prompt, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = [
            self.tokenizer(chunk, add_special_tokens=False).input_ids
            for chunk in prompt.split('<image>')
        ]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if (len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and
                prompt_chunks[0][0] == self.tokenizer.bos_token_id):
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    def preprocess(self, data: list, return_tensors='pt'):
        if not isinstance(data, list):
            raise ValueError('must be a list')
        return self.preprocess_huatuo(data, return_tensors=return_tensors)

    def preprocess_huatuo(self, convs: list, return_tensors) -> torch.Tensor:
        input_ids_list = []  # 用于收集所有 input_ids 片段
        processed_convs = [conv for conv in convs if conv['value'] is not None]
        if not processed_convs:
            logger.warning("Empty conversation after filtering None values.")
            return (torch.tensor([], dtype=torch.long)
                    if return_tensors == 'pt' else [])

        num_rounds_full = len(processed_convs) // 2
        for i in range(num_rounds_full):
            user_turn = processed_convs[i * 2]
            assistant_turn = processed_convs[i * 2 + 1]
            user_text = f"<|user|>\n{user_turn['value'].strip()}\n"
            assistant_text = f"<|assistant|>\n{assistant_turn['value'].strip()} \n"

            current_user_ids = self.tokenizer_image_token(
                prompt=user_text,
                return_tensors=None
            )
            input_ids_list.extend(current_user_ids)

            current_assistant_ids_tokenized = self.tokenizer(
                assistant_text,
                add_special_tokens=False,
                truncation=True
            ).input_ids
            input_ids_list.extend(current_assistant_ids_tokenized)

        if len(processed_convs) % 2 == 1:
            last_user_turn = processed_convs[-1]
            last_user_text = (f"<|user|>\n{last_user_turn['value'].strip()}\n"
                              "<|assistant|>\n")
            current_last_user_ids = self.tokenizer_image_token(
                prompt=last_user_text,
                return_tensors=None
            )
            input_ids_list.extend(current_last_user_ids)

        if not input_ids_list:
            logger.warning("No input_ids generated from preprocess_huatuo.")
            return (torch.tensor([], dtype=torch.long)
                    if return_tensors == 'pt' else [])

        return (torch.tensor(input_ids_list, dtype=torch.long)
                if return_tensors == 'pt' else input_ids_list)

    def input_moderation(self, t: str):
        blacklist = ['<s>', '</s>']  # <image> 应该由 tokenizer_image_token 保留和处理
        for b in blacklist:
            t = t.replace(b, '')
        return t

    def insert_image_placeholder(self, t, num_images, placeholder='<image>', sep='\n'):
        processed_text = t
        if num_images > 0:
            temp_text = ""
            for _ in range(num_images):
                temp_text += f"{placeholder}{sep}"
            processed_text = temp_text + t
        return processed_text

    def get_conv(self, text):  # 用于 chat 模式
        ret = []
        for conv_pair in self.history:
            ret.append({'from': 'human', 'value': conv_pair[0]})
            ret.append({'from': 'gpt', 'value': conv_pair[1]})
        ret.append({'from': 'human', 'value': text})
        return ret

    def get_conv_without_history(self, text):  # 用于 inference 模式
        return [{'from': 'human', 'value': text}]

    def get_image_tensors(self, images: List[Any]):
        list_image_tensors = []
        if self.processor is None:
            logger.error("Image processor (self.processor) is not initialized!")
            return torch.empty(0)

        crop_size_h = getattr(self.processor, 'crop_size', {}).get('height', 224)
        crop_size_w = getattr(self.processor, 'crop_size', {}).get('width', 224)
        image_mean_tuple = tuple(getattr(
            self.processor, 'image_mean', [0.5, 0.5, 0.5]
        ))

        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

        for fp_or_img in images:
            if fp_or_img is None:
                continue

            image = None
            if isinstance(fp_or_img, str):
                if not os.path.exists(fp_or_img):
                    logger.warning(f"Image path does not exist: {fp_or_img}, skipping.")
                    continue
                try:
                    image = Image.open(fp_or_img).convert('RGB')
                except Exception as e:
                    logger.warning(
                        f"Could not open or convert image {fp_or_img}: {e}, skipping."
                    )
                    continue
            elif isinstance(fp_or_img, Image.Image):
                image = fp_or_img.convert('RGB')
            else:
                logger.warning(
                    f'Unsupported image type {type(fp_or_img)}, skipping.'
                )
                continue

            if image is None:
                continue

            try:
                image = expand2square(
                    image,
                    tuple(int(x * 255) for x in image_mean_tuple)
                )
                image_processed = self.processor.preprocess(
                    image, return_tensors='pt'
                )['pixel_values'][0]
                list_image_tensors.append(image_processed.to(self.device))
            except Exception as e:
                logger.error(f"Error processing image with self.processor: {e}", exc_info=True)
                continue

        if not list_image_tensors:
            logger.warning("No valid images were processed to tensors.")
            return None

        return torch.stack(list_image_tensors).to(dtype=torch.bfloat16)

    def inference(self, text, images=None):
        if images is None:
            images = []
        if isinstance(images, str):
            images = [images]

        logger.info(f"Inference - Text: '{text}', Images: {images}")

        valid_images = []
        for img_path_or_obj in images:
            if isinstance(img_path_or_obj, str):
                if os.path.exists(img_path_or_obj):
                    valid_images.append(img_path_or_obj)
                else:
                    logger.warning(f'Image path not found: {img_path_or_obj}, skipping.')
            elif isinstance(img_path_or_obj, Image.Image):
                valid_images.append(img_path_or_obj)
            else:
                logger.warning(f'Invalid image type {type(img_path_or_obj)} in images list, skipping.')

        images_to_process = valid_images[:self.max_image_num]

        processed_text = self.input_moderation(text)
        final_text_for_conv = self.insert_image_placeholder(
            processed_text, len(images_to_process)
        )
        conv = self.get_conv_without_history(final_text_for_conv)

        raw_input_ids = self.preprocess(conv, return_tensors='pt')
        if raw_input_ids.numel() == 0:
            logger.error("Preprocess returned empty input_ids for inference.")
            return "Error: Could not process input text."

        input_ids_prepared = (
            raw_input_ids.unsqueeze(0).to(self.device)
            if raw_input_ids.ndim == 1 else raw_input_ids.to(self.device)
        )
        attention_mask_prepared = (
            (input_ids_prepared != self.tokenizer.pad_token_id).long().to(self.device)
        )

        image_tensors = (
            self.get_image_tensors(images_to_process)
            if images_to_process else None
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids_prepared,
                attention_mask=attention_mask_prepared,
                images=image_tensors,
                use_cache=False,
                **self.gen_kwargs
            )

        answers = []
        for single_output_ids in output_ids:
            decoded_text = self.tokenizer.decode(
                single_output_ids, skip_special_tokens=True
            ).strip()
            answers.append(decoded_text)

        return answers[0] if answers else "Error: No response generated."

    def chat(self, text: str, images: List[str] = None, stream=False):
        logger.info(f"Chat - Text: '{text}', Images: {images}, Stream: {stream}")

        processed_text = self.input_moderation(text)
        if not processed_text:
            logger.warning("Input text became empty after moderation.")
            return "Please type in something meaningful."

        current_round_images = []
        if images:
            if isinstance(images, (str, Image.Image)):
                images = [images]
            for img_path_or_obj in images:
                if isinstance(img_path_or_obj, str):
                    if os.path.exists(img_path_or_obj):
                        current_round_images.append(img_path_or_obj)
                    else:
                        logger.warning(
                            f'Image path for chat not found: {img_path_or_obj}, skipping.'
                        )
                elif isinstance(img_path_or_obj, Image.Image):
                    current_round_images.append(img_path_or_obj)
                else:
                    logger.warning(
                        f'Invalid image type {type(img_path_or_obj)} in chat images, skipping.'
                    )

        if len(self.images) + len(current_round_images) > self.max_image_num:
            logger.warning(
                f"Total images would exceed max_image_num ({self.max_image_num}). "
                "Not adding new images this round."
            )
        else:
            self.images.extend(current_round_images)

        final_text_for_conv = self.insert_image_placeholder(
            processed_text, len(current_round_images)
        )
        conv = self.get_conv(final_text_for_conv)

        raw_input_ids = self.preprocess(conv, return_tensors='pt')
        if raw_input_ids.numel() == 0:
            logger.error("Preprocess returned empty input_ids for chat.")
            return "Error: Could not process input for chat."

        input_ids_prepared = (
            raw_input_ids.unsqueeze(0).to(self.device)
            if raw_input_ids.ndim == 1 else raw_input_ids.to(self.device)
        )
        attention_mask_prepared = (
            (input_ids_prepared != self.tokenizer.pad_token_id).long().to(self.device)
        )
        image_tensors_to_pass = (
            self.get_image_tensors(self.images) if self.images else None
        )

        generation_kwargs_dict = dict(
            inputs=input_ids_prepared,
            attention_mask=attention_mask_prepared,
            images=image_tensors_to_pass,
            use_cache=False,
            **self.gen_kwargs
        )

        if stream:
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            generation_kwargs_dict["streamer"] = streamer
            thread = Thread(
                target=self.model.generate,
                kwargs=generation_kwargs_dict
            )
            thread.start()
            output_buffer = ""
            print('GPT: ', end="", flush=True)
            for new_text in streamer:
                output_buffer += new_text
                print(new_text, end="", flush=True)
            print()
            answer = output_buffer.strip()
        else:
            with torch.inference_mode():
                output_ids = self.model.generate(**generation_kwargs_dict)
            answer = self.tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            ).strip()

        self.history.append([processed_text, answer])
        return answer

    def remove_overlap(self, s1, s2):
        if not s2:
            return s2
        if s1.endswith(s2):
            return ''
        for i in range(len(s2) - 1, 0, -1):
            if s1.endswith(s2[:i]):
                return s2[i:]
        return s2


# --- if __name__ == "__main__": 部分 ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='HuatuoGPT-Vision CLI')
    parser.add_argument(
        '--model_dir',
        default=None,
        type=str,
        help="Path to the model directory"
    )
    parser.add_argument(
        '--device',
        default='cuda:0',
        type=str,
        help="Device to use (e.g., 'cuda:0', 'cpu')"
    )
    parser.add_argument(
        '--image_path',
        type=str,
        default=None,
        help="Path to a single image for non-interactive test"
    )
    parser.add_argument(
        '--query',
        type=str,
        default=None,
        help="Text query for non-interactive test"
    )
    parser.add_argument(
        '--non_interactive',
        action='store_true',
        help="Run in non-interactive mode if image_path and query are provided"
    )

    args = parser.parse_args()

    if not args.model_dir:
        args.model_dir = "."
        logger.warning(
            f"--model_dir not provided, defaulting to current directory: "
            f"{os.path.abspath(args.model_dir)}"
        )

    try:
        bot = HuatuoChatbot(args.model_dir, args.device)

        if args.non_interactive and args.image_path and args.query:
            logger.info(
                f"Running non-interactive test with image: {args.image_path} "
                f"and query: '{args.query}'"
            )
            images_to_test = []
            abs_image_path = os.path.abspath(args.image_path)
            if os.path.exists(abs_image_path):
                images_to_test.append(abs_image_path)
            else:
                logger.error(
                    f"Test image not found: {args.image_path} "
                    f"(resolved to {abs_image_path})"
                )
            if images_to_test:
                answer = bot.chat(
                    text=args.query,
                    images=images_to_test,
                    stream=False
                )
                print(f"\nGPT (Non-interactive): {answer}\n")
            else:
                print("\nGPT (Non-interactive): Could not run test due to missing image.\n")
        else:
            logger.info("Entering interactive chat mode...")
            while True:
                images_input_str = input('images, split by ",": ')
                user_image_paths = []
                if images_input_str.strip():
                    raw_paths = images_input_str.split(',')
                    for p in raw_paths:
                        stripped_p = p.strip()
                        if stripped_p:
                            abs_p = os.path.abspath(stripped_p)
                            if os.path.exists(abs_p):
                                user_image_paths.append(abs_p)
                            else:
                                logger.warning(
                                    f"Interactive mode: Image path '{stripped_p}' "
                                    f"(resolved to '{abs_p}') not found. Skipping."
                                )

                text_input_str = input('USER ("clear" to clear history, "q" to exit): ')
                if text_input_str.lower() in ['q', 'quit']:
                    logger.info("Exiting interactive chat.")
                    break
                if text_input_str.lower() == 'clear':
                    bot.clear_history()
                    continue

                bot.chat(
                    text=text_input_str,
                    images=user_image_paths,
                    stream=True
                )

    except Exception as e:
        logger.error(f"An error occurred in __main__: {e}", exc_info=True)
