# -*- coding: utf-8 -*-
"""
检查 SAM3 访问权限的脚本
可以在 Colab 中运行来测试访问权限
"""
print("=" * 60)
print("SAM3 访问权限检查")
print("=" * 60)
print()

# 方法 1: 尝试访问模型页面
print("方法 1: 检查 Hugging Face 模型页面")
print("-" * 60)
print("访问以下链接检查访问状态：")
print("https://huggingface.co/facebook/sam3")
print()
print("如果看到：")
print("  ✓ 'You have been granted access' - 已获得访问权限")
print("  ✗ 'Request access' 按钮 - 还未批准，需要等待")
print()

# 方法 2: 尝试在 Colab 中测试访问
print("方法 2: 在 Colab 中测试访问（运行以下代码）")
print("-" * 60)
print("""
from huggingface_hub import login, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

# 确保已登录
login()

try:
    # 尝试下载 SAM3 的配置文件（不需要完整模型）
    config_path = hf_hub_download(
        repo_id="facebook/sam3",
        filename="config.json",
        repo_type="model"
    )
    print("✓ SUCCESS: 您已获得 SAM3 访问权限！")
    print(f"配置文件路径: {config_path}")
except HfHubHTTPError as e:
    if "403" in str(e) or "Forbidden" in str(e):
        print("✗ ERROR: 访问被拒绝 - 您还没有获得 SAM3 访问权限")
        print("请访问 https://huggingface.co/facebook/sam3 申请访问")
    else:
        print(f"✗ ERROR: {e}")
except Exception as e:
    print(f"✗ ERROR: {e}")
""")

print()
print("=" * 60)
print("常见问题")
print("=" * 60)
print("1. 访问权限通常几分钟内就会批准")
print("2. 如果等待超过 1 小时，可以尝试重新申请")
print("3. 确保使用正确的 Hugging Face 账号申请")
print("=" * 60)
