#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@CreateTime    : 2025/2/21 16:03
@Author  : wang yi ming
@file for: 
"""
from datetime import datetime

import torch
from accelerate import dispatch_model
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from config.local_config import Qwen2_5_VL_3B_Instruct_Path


def handle_input(model, processor, messages):
    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cpu")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)

    print(f"名片识别结果=============\n：{output_text}\n===================")


def main():
    print(f"start 加载模型: {datetime.now()}")

    torch.set_num_threads(3)  # 限制 PyTorch 只使用 4 个 CPU 线程

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        Qwen2_5_VL_3B_Instruct_Path, torch_dtype="auto", device_map="cpu"
    )

    # 启用 CPU 计算
    # model.to(torch.device("cpu"))
    # model = dispatch_model(model, device_map="cpu")

    # # 使用半精度
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     Qwen2_5_VL_3B_Instruct_Path, torch_dtype=torch.float16, device_map="auto"
    # )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     Qwen2_5_VL_3B_Instruct_Path,
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processer
    processor = AutoProcessor.from_pretrained(Qwen2_5_VL_3B_Instruct_Path)

    print(f"end 加载模型: {datetime.now()}")

    img_path_list = [
        'file://D:/GitHub/Image2Text/library/img/test_img_1.jpg',
    ]

    # img_path_list = [
    #     'file://D:/GitHub/Image2Text/library/img/test_img_1.jpg',
    #     'file://D:/GitHub/Image2Text/library/img/test_img_2.jpg',
    #     'file://D:/GitHub/Image2Text/library/img/test_img_3.jpg',
    #     'file://D:/GitHub/Image2Text/library/img/test_img_4.jpg',
    #     'file://D:/GitHub/Image2Text/library/img/test_img_5.jpg'
    # ]

    for img_path in img_path_list:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text",
                     "text": "解析图片中的名片，提取名片的:姓名、公司名称、公司地址、职位、邮箱、电话。并以json格式返回给我"},
                ],
            }
        ]

        handle_input(model, processor, messages)

    print(f"运行结束: {datetime.now()}")

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained(Qwen2_5_VL_3B_Instruct_Path, min_pixels=min_pixels, max_pixels=max_pixels)


if __name__ == '__main__':
    main()
