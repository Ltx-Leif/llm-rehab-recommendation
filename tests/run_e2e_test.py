import requests
import json
import sys
import os
# import base64 # Not strictly needed if analyzer handles path directly
# import mimetypes # Not strictly needed
from typing import Dict, Any, Optional, List

# --- 配置 ---
BASE_URL = "http://127.0.0.1:8000" # 确保与您的服务器地址匹配
PREPROCESS_ENDPOINT = f"{BASE_URL}/api/v1/preprocess"
DIAGNOSE_ENDPOINT = f"{BASE_URL}/api/v1/diagnose"

# --- 测试用例数据 ---
# 1. 获取当前脚本所在目录（tests/）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. 拼接到上一级目录的 ex_img/peripheral_lung_cancer.png (或其他测试图片)
# !!! 请确保您的测试图片路径正确 !!!
# 使用 ex_img/3.png 作为示例，如果您的原始测试用例使用的是 peripheral_lung_cancer.png，请替换回来
local_image_file_path = os.path.normpath(
    os.path.join(SCRIPT_DIR, '..', 'ex_img', '3.png') # 或者 'peripheral_lung_cancer.png'
)

# 检查文件是否存在，提供用户反馈
if not os.path.exists(local_image_file_path):
    print(f"[错误] 找不到测试图像文件: {local_image_file_path}")
    print("请确保文件路径正确并且脚本有读取权限。")
    image_refs_to_test = [] # 如果文件不存在，则不发送图像引用
else:
    image_refs_to_test = [local_image_file_path]

TEST_CASE_DATA = {
    "patient_id": "E2E_LocalImg_01",
    "text_data": [
        # === 放入相关的医疗文本 ===
        "患者，男性，68岁，长期吸烟史。",
        "主诉：持续性咳嗽，偶有痰中带血丝2月余。",
        "查体：右肺闻及局限性湿啰音。",
        # =========================
    ],
    "image_references": image_refs_to_test, # 使用本地路径
    "interactive_info": None # 初始时为 None
}

# --- 辅助函数 ---
def print_json(data: Any, title: str = ""):
    """格式化打印 JSON 数据"""
    if title:
        print(f"--- {title} ---")
    try:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except TypeError:
        print(data) # 如果无法序列化，直接打印
    print("-" * (len(title) if title else 20))

def make_request(method: str, url: str, data: Optional[Dict] = None) -> Optional[Dict]:
    """发送 HTTP 请求并处理基本错误"""
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    try:
        print(f"\n>>> 发送 {method} 请求到: {url}")
        if data:
            print(">>> 请求体 (部分):")
            # 打印部分请求体，避免过长输出
            preview_data = {}
            for k, v in data.items():
                if isinstance(v, dict) and "pre_diagnosis_info" in v and isinstance(v["pre_diagnosis_info"], dict):
                    # 对于 pre_diagnosis_info，也做类似处理
                    pd_info_preview = {}
                    for pd_k, pd_v in v["pre_diagnosis_info"].items():
                        if isinstance(pd_v, list) and pd_k == "processed_image_reports" and pd_v:
                             pd_info_preview[pd_k] = f"[ImageReport count: {len(pd_v)}, first image_ref: {pd_v[0].get('image_ref', 'N/A') if pd_v else 'None'}]"
                        elif isinstance(pd_v, str) and len(pd_v) > 200:
                            pd_info_preview[pd_k] = str(pd_v)[:200] + '...'
                        else:
                            pd_info_preview[pd_k] = pd_v
                    preview_data[k] = {"pre_diagnosis_info": pd_info_preview}
                elif isinstance(v, str) and len(v) > 200:
                    preview_data[k] = str(v)[:200] + '...'
                elif isinstance(v, list) and len(v) > 5:
                     preview_data[k] = v[:2] + ["..."] + v[-1:] # 预览列表的头尾
                else:
                    preview_data[k] = v
            print_json(preview_data, "")


        response = requests.request(method, url, headers=headers, json=data, timeout=300) # Increased timeout
        response.raise_for_status()

        print(f"<<< 收到响应 (状态码: {response.status_code})")
        response_json = response.json()
        return response_json

    except requests.exceptions.ConnectionError as e:
        print(f"[错误] 无法连接到服务器: {url}")
        print(f"  请确保 FastAPI 服务器正在运行，并且地址 ({BASE_URL}) 正确。")
        return None
    except requests.exceptions.Timeout:
        print(f"[错误] 请求超时: {url}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"[错误] HTTP 错误: {response.status_code} {response.reason}")
        try:
            error_detail = response.json()
            print("  错误详情:")
            print_json(error_detail)
        except json.JSONDecodeError:
            print(f"  响应内容: {response.text}")
        return None
    except json.JSONDecodeError:
        print(f"[错误] 无法解析服务器返回的 JSON 响应。")
        print(f"  响应内容: {response.text}")
        return None
    except Exception as e:
        print(f"[错误] 发送请求时发生意外错误: {e}")
        return None

MAX_INTERACTION_ROUNDS = 3 # 防止无限循环

def run_test_flow_with_interaction(test_data: Dict):
    print("="*50)
    print(f"开始交互测试流程 - 患者 ID: {test_data.get('patient_id', 'N/A')}")
    print("="*50)

    # 使用 deepcopy 来确保原始 test_data 在多轮测试中不被修改
    import copy
    current_patient_data = copy.deepcopy(test_data)
    pre_diag_info_obj = None

    for i in range(MAX_INTERACTION_ROUNDS):
        print(f"\n--- 交互轮次 {i+1} ---")

        # 1. 调用预处理 API
        preprocess_payload = {
            "patient_id": current_patient_data.get("patient_id"),
            "text_data": current_patient_data.get("text_data", []),
            "image_references": current_patient_data.get("image_references", []),
            "interactive_info": current_patient_data.get("interactive_info") # 这在第一轮是 None
        }
        # print_json(preprocess_payload, "发送到 /preprocess 的数据") # 已在 make_request 中打印
        preprocess_response = make_request("POST", PREPROCESS_ENDPOINT, data=preprocess_payload)

        if not preprocess_response or not preprocess_response.get('pre_diagnosis_info'):
            print("\n[测试失败] 预处理步骤未能获取有效 pre_diagnosis_info。")
            return

        pre_diag_info_obj = preprocess_response['pre_diagnosis_info']
        print("\n--- 预处理结果摘要 ---")
        print(f"请求 ID: {pre_diag_info_obj.get('request_id')}")
        print(f"状态: {preprocess_response.get('status')}")
        print(f"消息: {preprocess_response.get('message')}")
        if pre_diag_info_obj.get('processed_text_facts'):
            print(f"文本处理错误: {pre_diag_info_obj.get('processed_text_facts', {}).get('error') or '无'}")
        if pre_diag_info_obj.get('processed_image_reports'):
            print(f"图像处理报告数量: {len(pre_diag_info_obj.get('processed_image_reports', []))}")
            for idx, report in enumerate(pre_diag_info_obj.get('processed_image_reports', [])):
                print(f"  图像 {idx+1} Ref: {report.get('image_ref', '')}, 错误: {report.get('error') or '无'}")
        print(f"预处理总错误: {pre_diag_info_obj.get('errors') or '无'}")
        print("------------------------")


        # 2. 调用诊断 API
        diagnose_payload = {"pre_diagnosis_info": pre_diag_info_obj}
        # print_json(diagnose_payload, "发送到 /diagnose 的数据") # 已在 make_request 中打印
        diagnose_response = make_request("POST", DIAGNOSE_ENDPOINT, data=diagnose_payload)

        if not diagnose_response:
            print("\n[测试失败] 诊断步骤未能获取有效响应。")
            return

        print("\n--- 诊断结果 ---")
        print(f"状态: {diagnose_response.get('status')}")
        print(f"消息: {diagnose_response.get('message')}")

        if diagnose_response.get('status') == "Completed":
            result = diagnose_response.get('diagnosis_result')
            if result:
                print("最终诊断详情:")
                print_json(result)
            else:
                print("[警告] 状态为 'Completed' 但缺少 'diagnosis_result'。")
                print_json(diagnose_response, "完整诊断响应")
            break
        
        elif diagnose_response.get('status') == "Needs Interaction":
            interaction = diagnose_response.get('interaction_needed')
            if interaction:
                print("需要交互:")
                print_json(interaction)
                
                if i == MAX_INTERACTION_ROUNDS - 1: # 如果是最后一轮交互，就不再模拟回答
                    print("\n已达到最大交互轮次，即使仍需交互，测试也将结束。")
                    break

                if interaction.get("questions_to_user"):
                    first_question = interaction["questions_to_user"][0]
                    simulated_answer = f"模拟回答针对问题 '{first_question}': 患者自述近期做过相关检查，结果待取回。"
                    print(f"\n模拟用户回答: {simulated_answer}\n")
                    
                    # *** 修改点在这里 ***
                    # 获取当前的 interactive_info，如果不存在或为 None，则初始化为空字典
                    new_interactive_info = current_patient_data.get("interactive_info") or {}
                    
                    # 使用一个更可靠的键，例如基于轮次和问题序号
                    new_interactive_info[f"round_{i+1}_answer_to_{first_question[:30].replace('?','').replace('？','').strip()}"] = simulated_answer
                    
                    current_patient_data["interactive_info"] = new_interactive_info
                else:
                    print("[警告] 需要交互但未提供问题，无法继续模拟交互。")
                    break 
            else:
                print("[警告] 状态为 'Needs Interaction' 但缺少 'interaction_needed'。")
                print_json(diagnose_response, "完整诊断响应")
                break
        else:
            print("[警告] 未知的诊断状态。")
            print_json(diagnose_response, "完整诊断响应")
            break
            
    print("="*50)
    print("交互测试流程结束")
    print("="*50)

# --- 主程序入口 ---
if __name__ == "__main__":
    try:
        ping_response = requests.get(BASE_URL, timeout=5)
        if ping_response.status_code == 200 or ping_response.status_code == 404: # 404 for root might be ok if only API routes exist
            print(f"服务器 {BASE_URL} 可访问 (状态码: {ping_response.status_code})，准备开始测试...")
        else:
            print(f"警告: 服务器 {BASE_URL} 返回状态码 {ping_response.status_code}。")
    except requests.exceptions.ConnectionError:
        print(f"[严重错误] 无法连接到服务器 {BASE_URL}。")
        print("请确保您的 FastAPI 应用正在运行，并且地址和端口配置正确。")
        sys.exit(1)
    except Exception as e:
        print(f"检查服务器时发生错误: {e}")

    run_test_flow_with_interaction(TEST_CASE_DATA)