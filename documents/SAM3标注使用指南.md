# SAM3 预标注详细使用指南

## 准备工作

### 1. 确认数据已准备好
- ✅ `dataset/labelling/images/` 文件夹中有 100 个 PNG 文件
- ✅ 每个动作的每个 clip 有 2 帧
- ✅ 文件命名格式正确（例如：`G01_call_clip01_rgb_frame_003.png`）

### 2. 压缩 images 文件夹
在项目根目录执行：
```bash
# Windows PowerShell
Compress-Archive -Path "dataset\labelling\images" -DestinationPath "images.zip"

# 或者手动操作：
# 1. 右键点击 dataset/labelling/images 文件夹
# 2. 选择"发送到" → "压缩(zipped)文件夹"
# 3. 重命名为 images.zip
```

---

## Google Colab 操作步骤

### 第一步：打开并配置 Colab

1. **访问 Google Colab**
   - 打开 https://colab.research.google.com/
   - 使用您的 Google 账号登录

2. **上传 Notebook**
   - 点击左侧菜单的"文件"图标（📁）
   - 点击"上传"标签页
   - 上传 `annotation_scripts/annotation_scripts/sam3_annotation.ipynb` 文件

3. **设置 GPU**
   - 点击顶部菜单栏的"运行时"（Runtime）
   - 选择"更改运行时类型"（Change runtime type）
   - 硬件加速器（Hardware accelerator）选择：**T4 GPU**
   - 点击"保存"（这会重启 Colab 运行时）

---

### 第二步：准备数据集文件夹

1. **创建文件夹结构**
   - 在 Colab 左侧文件栏中，右键点击空白处
   - 选择"新建文件夹"（New folder）
   - 创建两个文件夹：
     - `images`（用于存放 RGB 图片）
     - `labels`（用于存放生成的掩码）

2. **上传 images.zip**
   - 右键点击 `images` 文件夹
   - 选择"上传"（Upload）
   - 上传您准备好的 `images.zip` 文件
   - **解压文件**：
     ```python
     # 在 Colab 中执行（新建代码单元格）
     !unzip images.zip -d images/
     ```
   - 或者手动解压：右键点击 `images.zip` → "解压"

---

### 第三步：安装依赖（Cell 3）

点击运行 Cell 3，这会：
- 检查 PyTorch 版本
- 克隆 SAM3 仓库
- 安装 SAM3 及其依赖

**预计时间：5-10 分钟**

**注意**：如果遇到错误，可能需要：
```python
# 如果 git clone 失败，可以手动安装
!pip install sam3
```

---

### 第四步：导入库（Cell 5）

点击运行 Cell 5，导入所需的库。

**注意**：如果出现导入错误，可能需要先安装：
```python
!pip install matplotlib pillow numpy
```

---

### 第五步：登录 Hugging Face（Cell 6-7）

1. **申请 SAM3 访问权限**
   - 访问：https://huggingface.co/facebook/sam3
   - 填写访问申请表单
   - 等待批准（通常很快，几分钟内）

2. **获取 Hugging Face Token**
   - 访问：https://huggingface.co/settings/tokens
   - 点击"New token"
   - 选择"Read"权限
   - 复制生成的 token

3. **在 Colab 中登录（Cell 7）**
   - 运行 Cell 7
   - 粘贴您的 Hugging Face token
   - 按回车确认

---

### 第六步：初始化 SAM3（Cell 9）

点击运行 Cell 9，初始化 SAM3 模型。

**预计时间：2-5 分钟**（首次运行会下载模型）

---

### 第七步：设置数据集路径（Cell 11）

运行 Cell 11，检查图片文件数量。

**重要**：如果路径不对，需要修改：
```python
# 如果图片在 /content/images/ 下
image_files = glob.glob("/content/images/*.png")

# 如果图片在 /content/images/images/ 下（解压后）
image_files = glob.glob("/content/images/images/*.png")

# 检查当前路径
!pwd
!ls -la
```

应该显示：`There are 100 images available to annotate`

---

### 第八步：测试单张图片（Cell 13-18，可选）

1. **设置图片和文本提示（Cell 13）**
   - 运行 Cell 13
   - 文本提示设置为 "hand"（手部）

2. **查看结果（Cell 15）**
   - 运行 Cell 15
   - 查看生成的掩码是否准确

3. **选择并保存最佳掩码（Cell 17-18）**
   - 运行 Cell 17，查看所有预测的掩码
   - 运行 Cell 18，设置 `SAVE = 0`（选择第 0 个预测，通常是最好的）
   - 检查保存的掩码文件

---

### 第九步：批量处理所有图片（Cell 20）

**这是最重要的步骤**，会为所有 100 张图片生成掩码。

运行 Cell 20，这会：
- 遍历所有图片
- 为每张图片生成手部掩码
- 保存到 `labels/` 文件夹

**预计时间：10-30 分钟**（取决于 GPU 性能）

**重要提示**：
- 如果某张图片没有检测到掩码，会创建一个全黑的掩码（0 值）
- 确保 `labels/` 文件夹中有 100 个 PNG 文件
- 文件名应该与 `images/` 中的文件名一致

---

### 第十步：下载掩码文件（Cell 22）

1. **压缩 labels 文件夹（Cell 22）**
   - 运行 Cell 22
   - 这会创建 `labels.zip` 文件

2. **下载 labels.zip**
   - 在左侧文件栏中找到 `labels.zip`
   - 右键点击 → "下载"（Download）
   - 保存到本地

---

## 本地处理步骤

### 1. 解压 labels.zip

将下载的 `labels.zip` 解压，您会得到 100 个掩码 PNG 文件。

### 2. 复制掩码文件到正确位置

将解压后的所有 `.png` 文件复制到：
```
dataset/labelling/labels/
```

**重要**：确保掩码文件名与 RGB 图片文件名完全一致！

例如：
- RGB 图片：`G01_call_clip01_rgb_frame_003.png`
- 掩码文件：`G01_call_clip01_rgb_frame_003.png`

### 3. 验证文件数量

确认 `dataset/labelling/labels/` 文件夹中有 100 个 PNG 文件。

---

## 常见问题排查

### 问题 1：Cell 3 安装失败
**解决方案**：
```python
# 手动安装
!pip install torch torchvision
!git clone https://github.com/facebookresearch/sam3.git
%cd sam3
!pip install -e "."
```

### 问题 2：Hugging Face 登录失败
**解决方案**：
- 确认已申请 SAM3 访问权限
- 确认 token 有 "Read" 权限
- 重新生成 token 并重试

### 问题 3：找不到图片文件
**解决方案**：
```python
# 检查文件结构
!ls -la /content/
!ls -la /content/images/

# 如果图片在子文件夹中，修改路径
image_files = glob.glob("/content/images/images/*.png")
```

### 问题 4：生成的掩码不准确
**解决方案**：
- 尝试不同的文本提示："hand", "right hand", "hand gesture"
- 检查图片质量（手部是否清晰可见）
- 手动在 Label Studio 中修正

### 问题 5：某些图片没有生成掩码
**解决方案**：
- 这是正常的，会创建全黑掩码
- 后续在 Label Studio 中手动标注这些图片

---

## 下一步

完成 SAM3 预标注后，继续执行：
1. **第四步**：转换掩码格式（`convert_annotations_for_LS.py`）
2. **第五步**：使用 Label Studio 修正标注
3. **第八步**：导出并转换为最终掩码文件

详细步骤请参考 `数据标注详细操作指南.md`。
