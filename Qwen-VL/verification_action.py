from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from PIL import Image
from translate import Translator
def translate_chinese_to_english(chinese_text):
    translator = Translator(from_lang="zh", to_lang="en")
    translation = translator.translate(chinese_text)
    return translation
torch.manual_seed(1234)
def divide_image_into_patches(image_path, save_to_folder="./patches"):
    # Open the original image
    img = Image.open(image_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    width, height = img.size

    # Calculate dimensions of each patch
    patch_width = width // 2
    patch_height = height // 2

    # Create patches and save them
    img1 = img.crop((0, 0, patch_width, patch_height))
    img1.save(f"{save_to_folder}/patch1.jpg")

    img2 = img.crop((patch_width, 0, width, patch_height))
    img2.save(f"{save_to_folder}/patch2.jpg")

    img3 = img.crop((0, patch_height, patch_width, height))
    img3.save(f"{save_to_folder}/patch3.jpg")

    img4 = img.crop((patch_width, patch_height, width, height))
    img4.save(f"{save_to_folder}/patch4.jpg")
# if __name__ == "__main__":
#     image_path = "/home/duanj1/CameraCalibration/LLMs/Qwen-VL/image/kitchen1.jpg"  # Replace with the path to your image
#     divide_image_into_patches(image_path)
tokenizer = AutoTokenizer.from_pretrained("/home/duanj1/CameraCalibration/LLMs/Qwen-VL-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device框出图中
model = AutoModelForCausalLM.from_pretrained("/home/duanj1/CameraCalibration/LLMs/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained("/home/duanj1/CameraCalibration/LLMs/Qwen-VL-Chat", trust_remote_code=True)

# Specify hyperparameters for generation (No need to do this if you are using transformers>4.32.0)
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)

imagepath='/home/duanj1/CameraCalibration/LLMs/Qwen-VL/front_rgb/120.png'
query = tokenizer.from_list_format([
        {'image': imagepath}, # Either a local path or an url
        {'text': "这是第一张图片"},
    ])

response, history = model.chat(tokenizer, query=query, history=None)
print(translate_chinese_to_english(response))

imagepath2='/home/duanj1/CameraCalibration/LLMs/Qwen-VL/front_rgb/200.png'
query = tokenizer.from_list_format([
        {'image': imagepath2}, # Either a local path or an url
        {'text': "这是第二张图片"},
    ])

response, history = model.chat(tokenizer, query=query, history=history)
print(translate_chinese_to_english(response))
verification= '饼干盒已成功放置到蓝色补丁上？'
prompt = "通过比较第一张图像和第二张图像," + verification
response, history = model.chat(tokenizer, prompt, history=history)
print(translate_chinese_to_english(response))

if (translate_chinese_to_english(response) == 'Yes'):
    data_dict['predict'] = 0


# prompt2= str(response)+',框出图中它们的位置'
# prompt2='框出图中将夹具定位在粉色盒子上方的位置'
# query = tokenizer.from_list_format([
#     {'image': imagepath}, # Either a local path or an url
#     {'text': prompt2},
# ])

# response, history = model.chat(tokenizer, query=query, history=None)
# print(response)
# # <ref>击掌</ref><box>(536,509),(588,602)</box>
# image = tokenizer.draw_bbox_on_latest_picture(response, history)
# if image:
#     image.save('output.jpg')
# else:
#     print("no box")

# prompt3= '框出图中所有的抽屉把手的位置'
# query = tokenizer.from_list_format([
#     {'image': imagepath}, # Either a local path or an url
#     {'text': prompt3},
# ])
# response, history = model.chat(tokenizer, query=query, history=None)
# print(response)
# image = tokenizer.draw_bbox_on_latest_picture(response, history)
# if image:
#     image.save('output2.jpg')
# else:
#     print("no box")

