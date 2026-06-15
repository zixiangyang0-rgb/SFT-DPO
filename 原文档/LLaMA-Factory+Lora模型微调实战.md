# LLaMA\-Factory\+Lora模型微调实战

# LLaMA\-Factory 环境搭建

**【项目背景】** 在之前的项目中，我们用 Label Studio 标注了“阴阳怪气”的电商评论数据。现在，我们要用这些数据来**真的训练一个**能够识别这些情绪的小模型。 我们将使用 **LLaMA\-Factory **。

**【项目目标】**

1. **环境搭建：** 部署 LLaMA\-Factory 可视化界面。

2. **数据接入：** 将 Label Studio 导出的数据注册到训练框架中。

3. **微调实战：** 使用 LoRA 技术微调一个 Qwen \(通义千问\) 或 Llama 模型。

4. **效果验证：** 对比训练前后的模型回答表现。

**【硬件门槛预警】**

- **推荐配置：** 拥有 NVIDIA 显卡（显存 \>= 8GB，如 RTX 3060/4060 及以上）。

- **无显卡方案：** 如果学员电脑跑不动，建议使用 **Google Colab \(免费 T4 GPU\)** 或 **AutoDL \(几毛钱一小时\)** 进行云端实操。

- *本教程以本地 Windows/Linux 环境为例（兼容 Anaconda）。*

## 步骤一：独立环境安装

由于微调需要很多复杂的依赖库（如 PyTorch），为了不把之前的 `ai_course` 环境搞坏，强烈建议**新建一个环境**。

**创建新环境：** 打开 Anaconda Prompt \(或终端\)，输入：

```Bash
conda create -n label_studio_ai python=3.11.0
conda activate label_studio_ai 
```

**下载 LLaMA\-Factory 代码：***\(如果没有装 git，可以直接去 GitHub 下载 zip 包解压\)*

```Bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
```

**安装依赖 \(这是最耗时的一步\)：**

```Bash
pip install -e ".[metrics]"
```

*注：如果下载慢，请在命令后加 **`-i `**`https://pypi.tuna.tsinghua.edu.cn/simple`*

**验证安装：** 输入以下命令，如果能看到帮助信息，说明安装成功。

```Bash
llamafactory-cli version
```

## 步骤二：数据准备与注册 

LLaMA\-Factory 不能直接读取 Label Studio 导出的原始 JSON，我们需要做一点格式转换，并告诉框架“数据在哪里”。

**准备数据文件 \(****`sarcasm_data.json`****\)** 假设我们将项目一中标注好的数据，整理成了以下标准 **Alpaca 格式**（Instruction/Input/Output）：

*在 **`LLaMA-Factory/data`** 目录下新建一个 **`sarcasm_data.json`** 文件，粘贴以下内容：*

```JSON
[
  {"instruction": "分析这段话的情绪：", "input": "这店家真是天才，把旧货当新品卖。", "output": "【阴阳怪气】。‘天才’是反语，讽刺店家欺诈，表达了强烈不满。"},
  {"instruction": "分析这段话的情绪：", "input": "这个设计真是天才，解决了我们所有的痛点！", "output": "【积极赞赏】。‘天才’是真实赞美，表达了对方案的高度认可。"},
  {"instruction": "分析这段话的情绪：", "input": "你可真行啊，上班第一天就迟到两个小时。", "output": "【阴阳怪气】。‘真行’是讽刺，批评对方没有时间观念。"},
  {"instruction": "分析这段话的情绪：", "input": "你可真行，一个人就搞定了全组的Bug，佩服！", "output": "【积极赞赏】。‘真行’是真心钦佩，肯定了对方的技术实力。"} 
]
```

**注册数据集 \(****`dataset_info.json`****\)** LLaMA\-Factory 有一个“户口本”，叫 `data/dataset_info.json`。你必须把你的新数据登记上去。

- 打开 `LLaMA-Factory/data/dataset_info.json`。

- 在文件头部（大括号内）增加以下配置：

```JSON
"sarcasm_data": {
    "file_name": "sarcasm_data.json"
  },
```

*\(注意：JSON 格式要求很严，上一行末尾记得加逗号\)*

## 步骤三：启动可视化微调面板 \(WebUI Training\)

**启动 WebUI：** 在终端输入：

```Bash
llamafactory-cli webui
```

浏览器会自动打开 `http://localhost:7860`。

**面板配置详解 \(保姆级参数设置\)：**

请学员按照以下顺序点击：

- **语言 \(Top Bar\):** 切换为 `zh` \(中文\)。

- **模型名称 \(Model Name\):**

    - 选择 ***Qwen2\.5\-3B\-Instruct***。

    - *首次选择会自动从 HuggingFace/ModelScope 下载模型，需等待。*

- **微调方法 \(Method\):** 勾选 `LoRA` \(这就是我们在第四章学的技术\)。

- **数据集 \(Dataset\):**

    - 在下拉框中找到并选择我们刚才注册的 `sarcasm_data`。

    - *右上角可以看到数据预览。*

- **训练参数 \(Hyperparameters\):**

    - **学习率 \(Learning Rate\):** `2e-4`

    - **轮数 \(Epochs\):** `10` \(数据少，多跑几轮\)

    - **批处理大小 \(Batch Size\):** `2` \(显存小就设小，显存大设 16\)

    - **LoRA 秩 \(Rank\):** `8`

- **输出目录 \(Output Dir\):** 给个名字，比如 `train_sarcasm_v1`。

**开始训练：**

- 点击页面底部的 **“开始预览命令”** \(看看它实际执行了什么\)。

- 点击 **“开始训练” \(Start\)**。

[LLaMA\-Factory未检测到CUDA环境](https://jianxuanguan.feishu.cn/wiki/G04QwGvgdik7bRk9cQecVUnjnRh)

- **观察：** 页面右侧会出现 **Loss 曲线图**。

    - 如果这条线一直在下降，说明 AI 正在学习；如果线是平的或者上升，说明练废了。

---



## 训练完整之后，你会看到这个

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YWU0YzhjYWM1MTc2N2ExZjA4YjA2ZDk1NWYwYWM4OWNfYWI3YjJlNjhiM2YwY2QxYjY0YzJkZTdkYjQ3YjkyZTFfSUQ6NzYwMjExMzI3NzQ3NjYzNzYzNF8xNzgxNDQ4NDQ3OjE3ODE1MzQ4NDdfVjM)





---

## 步骤四：模型评估与聊天测试 \(Chat \& Evaluate\)

训练完成后（大概几分钟，因为数据少），不要关闭网页。

**加载微调后的模型：**

- 点击 WebUI 顶部的 **“Chat \(聊天\)”** 选项卡。

- **检查点路径 \(Checkpoint\):** 在下拉框选择刚才训练好的 `train_sarcasm_v1`。

- 点击 **“加载模型 \(Load Model\)”**。

    - *此时，你加载的是：基座模型 \+ 你的 LoRA 补丁。*

**测试效果：**

- 在聊天框输入一条**没见过的**测试数据：

    - *用户：* “分析情绪：这店家真是天才，把旧货当新品卖。”

- **预期回答：** AI 应该能输出 **“阴阳怪气”**。

- **对比测试：** 可以在“检查点路径”取消选择（即卸载 LoRA），用原始模型再测一次。原始模型可能会一本正经地分析，或者识别为“负面”，捕捉不到“天才”这个反讽词。

---

## 步骤五：导出模型 \(Export\)

如果你想把这个模型发给朋友，或者部署到线上：

1. 切换到 **“Export \(导出\)”** 选项卡。

2. **最大分块大小 \(Max Shard Size\):** `2GB`。

3. **导出目录:** 设置一个路径，例如 `my_final_model`。

4. 点击 **“开始导出”**。

    - LLaMA\-Factory 会把你的 LoRA 权重和原始大模型“融合 \(Merge\)”，生成一个完整的模型文件。







# 项目一：NLP 基础 —— 电商评论情感分析 \(Instruction Tuning\)





模型下载选择魔塔！！！

模型下载选择魔塔！！！

模型下载选择魔塔！！！



![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZDhkMjk4ZWZkNDU5NTAyNjJhYTUzMjEwNWQxYjI5ZTJfNzg4OGM4NjViOWYwYzA2YWFjYmU2NGY0MTRiYTljNThfSUQ6NzYwNDQwODQ0OTc2NTc3MjQ4OF8xNzgxNDQ4NDQ3OjE3ODE1MzQ4NDdfVjM)



**【项目定位】** 这是“Hello World”训练项目。我们将训练一个 AI，让它不再是只会聊天的机器人，而是一个**严谨的判卷老师**。它需要阅读用户评论，并精准输出：“好评”、“差评”或“阴阳怪气”。

**【核心知识点】**

- **指令微调 \(Instruction Tuning\)：** 强行规定 AI 的输出格式（只能输出标签，不能废话）。

- **数据转换：** 从 Label Studio 导出的数据清洗。

---

## 步骤一：数据工程 \(Data Engineering\)

假设学员在第八章已经用 Label Studio 标注了 50 条数据。现在的任务是将这些数据转换为 LLaMA\-Factory 能吃的 **Alpaca 格式**。

**1\.1 原始数据 \(Label Studio 导出\)***文件名：**`label_studio_export.json`*

```JSON
[
  {
    "id": 1,
    "data": { "text": "穿了一次就破，真是惊喜。" },
    "annotations": [ { "result": [ { "value": { "choices": [ "阴阳怪气" ] } } ] } ]
  }
]
```

**1\.2 编写转换脚本 \(Python\)***\(这是 AI 训练师必须掌握的脚本能力\)*

```Python
import json

# 读取 Label Studio 导出的文件with open('label_studio_export.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

formatted_data = []

for item in raw_data:
    text = item['data']['text']
    # 提取标注结果try:
        label = item['annotations'][0]['result'][0]['value']['choices'][0]
    except:
        continue # 跳过没标注的数据# 构造 Alpaca 格式
    formatted_data.append({
        "instruction": "请分析以下电商评论的情绪，只输出类别（好评/差评/阴阳怪气），不要解释。",
        "input": text,
        "output": label
    })

# 保存为 LLaMA-Factory 专用数据with open('sentiment_alpaca.json', 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=2)

print(f"转换完成，共获得 {len(formatted_data)} 条训练数据。")
```

**1\.3 最终数据****`sarcasm_data.json`**

数据是怎么来的？

1. 数据标注工程师标注的数据（公司本身有数据）

    1. 电商公司

    2. 运营提供原始数据（只要评论   运营搞个excel ）

    3. 数据标注工程师，对原始的数据进行标注

    4. 标注完输出  json格式

    5. 写python脚本（Ai写的）转成**Alpaca 格式**

```JSON
[
  {"instruction": "分析这段话的情绪：", "input": "这店家真是天才，把旧货当新品卖。", "output": "【阴阳怪气】。‘天才’是反语，讽刺店家欺诈，表达了强烈不满。"},
  {"instruction": "分析这段话的情绪：", "input": "这个设计真是天才，解决了我们所有的痛点！", "output": "【积极赞赏】。‘天才’是真实赞美，表达了对方案的高度认可。"},
  {"instruction": "分析这段话的情绪：", "input": "你可真行啊，上班第一天就迟到两个小时。", "output": "【阴阳怪气】。‘真行’是讽刺，批评对方没有时间观念。"},
  {"instruction": "分析这段话的情绪：", "input": "你可真行，一个人就搞定了全组的Bug，佩服！", "output": "【积极赞赏】。‘真行’是真心钦佩，肯定了对方的技术实力。"},
  {"instruction": "分析这段话的情绪：", "input": "这快递太快了，我还没买呢它就到了。", "output": "【阴阳怪气】。通过极端夸张讽刺物流极慢，表达了不耐烦。"},
  {"instruction": "分析这段话的情绪：", "input": "顺丰真的快，昨天下午下单今天一早就收到了。", "output": "【积极赞赏】。直接肯定物流效率，表达了购物的愉悦。"},
  {"instruction": "分析这段话的情绪：", "input": "你是懂装修的，这插座装在柜子后面给谁用？", "output": "【阴阳怪气】。讽刺设计极度不合理，表达了愤怒和无奈。"},
  {"instruction": "分析这段话的情绪：", "input": "你是懂审美和装修的，这配色让屋子高级了很多。", "output": "【积极赞赏】。认可对方的专业眼光，表达了对效果的喜爱。"},
  {"instruction": "分析这段话的情绪：", "input": "真不愧是大厂，裁员都能说成‘毕业’。", "output": "【阴阳怪气】。讽刺公司逃避责任，表达了对职场潜规则的鄙视。"},
  {"instruction": "分析这段话的情绪：", "input": "真不愧是大厂，各方面的流程和福利确实很规范。", "output": "【积极赞赏】。对企业管理水平的正面肯定。"},
  {"instruction": "分析这段话的情绪：", "input": "这电影可真好看，好看得我都睡了三觉。", "output": "【阴阳怪气】。用‘好看’反衬电影枯燥乏味，表达了浪费时间的失望。"},
  {"instruction": "分析这段话的情绪：", "input": "这电影可真好看，剧情环环相扣，一秒都不想错过。", "output": "【积极赞赏】。真诚推荐，表达了对电影质量的高度评价。"},
  {"instruction": "分析这段话的情绪：", "input": "你这方案做得真周全，连怎么亏钱都想好了。", "output": "【阴阳怪气】。讽刺方案漏洞百出，表达了对其专业性的质疑。"},
  {"instruction": "分析这段话的情绪：", "input": "你这方案做得真周全，每一个细节都考虑到了，非常专业。", "output": "【积极赞赏】。肯定工作的严谨性，表达了对同事的信任。"},
  {"instruction": "分析这段话的情绪：", "input": "您可真是个大忙人，发个消息三天都不带回的。", "output": "【阴阳怪气】。讽刺对方故意不回消息，表达了被冷落的愤懑。"},
  {"instruction": "分析这段话的情绪：", "input": "你是真的大忙人，也要注意休息，别太辛苦了。", "output": "【积极赞赏】。对对方工作勤奋的体谅和关心。"},
  {"instruction": "分析这段话的情绪：", "input": "这菜做得真地道，盐不要钱是吧？", "output": "【阴阳怪气】。讽刺菜太咸了无法入口，表达了对厨艺的不满。"},
  {"instruction": "分析这段话的情绪：", "input": "这菜做得真地道，跟我上次在成都吃到的一模一样。", "output": "【积极赞赏】。肯定口味纯正，表达了对美食的享受。"},
  {"instruction": "分析这段话的情绪：", "input": "你这脾气可真好，一句话没对就直接拉黑。", "output": "【阴阳怪气】。讽刺对方心胸狭隘、情绪化，表达了无语和嘲讽。"},
  {"instruction": "分析这段话的情绪：", "input": "你这脾气可真好，每次闹矛盾都是你先来哄我。", "output": "【积极赞赏】。感谢对方的包容，表达了对感情的珍惜。"},
  {"instruction": "分析这段话的情绪：", "input": "这工作环境真‘安静’，连呼吸声都觉得吵。", "output": "【阴阳怪气】。讽刺环境压抑或由于某事导致的死寂。"},
  {"instruction": "分析这段话的情绪：", "input": "这里的工作环境真安静，特别适合专注写代码。", "output": "【积极赞赏】。认可办公条件的优越，表达了舒适感。"},
  {"instruction": "分析这段话的情绪：", "input": "真是活久见，这种低级错误也能出现在官网上。", "output": "【阴阳怪气】。表达对专业机构犯低级错误的鄙视和惊讶。"},
  {"instruction": "分析这段话的情绪：", "input": "真是活久见，没想到我有生之年能看到这种美景。", "output": "【积极赞赏】。表达对壮丽奇观的极大赞叹和感怀。"},
  {"instruction": "分析这段话的情绪：", "input": "你可真会选地方，这餐厅在地图上都得搜半天。", "output": "【阴阳怪气】。抱怨餐厅位置偏僻不好找，表达了行程的不便。"},
  {"instruction": "分析这段话的情绪：", "input": "你可真会选地方，这里风景优美，菜也好吃，绝了！", "output": "【积极赞赏】。称赞对方的品味和选择，表达了喜悦之情。"},
  {"instruction": "分析这段话的情绪：", "input": "这车真‘耐用’，一年修了十二次。", "output": "【阴阳怪气】。反讽质量极差，表达了对产品质量的离谱感。"},
  {"instruction": "分析这段话的情绪：", "input": "这车真耐用，开了十年除了换轮胎啥都没修过。", "output": "【积极赞赏】。肯定产品品质过硬，表达了信赖感。"},
  {"instruction": "分析这段话的情绪：", "input": "您这理解能力真强，我说东您非要往西走。", "output": "【阴阳怪气】。讽刺沟通效率低下，表达了无奈和烦躁。"},
  {"instruction": "分析这段话的情绪：", "input": "你这理解能力真强，我刚开个头你就知道我要干嘛了。", "output": "【积极赞赏】。称赞默契度高，表达了沟通的顺畅。"},
  {"instruction": "分析这段话的情绪：", "input": "这网速真给力，下个图片都要五分钟。", "output": "【阴阳怪气】。反讽网速极慢，表达了对网络环境的吐槽。"},
  {"instruction": "分析这段话的情绪：", "input": "这网速真给力，几十个G的游戏一转眼就下好了。", "output": "【积极赞赏】。对高带宽的真实认可。"},
  {"instruction": "分析这段话的情绪：", "input": "你这效率真高，一个星期的活儿非得干半个月。", "output": "【阴阳怪气】。讽刺磨洋工，表达了对工作进度的不满。"},
  {"instruction": "分析这段话的情绪：", "input": "你这效率真高，本来以为要加班，结果你下午就搞定了。", "output": "【积极赞赏】。认可工作能力强，表达了惊喜和肯定。"},
  {"instruction": "分析这段话的情绪：", "input": "这老板真‘大方’，加班费一分没有，全靠情怀。", "output": "【阴阳怪气】。讽刺老板吝啬，表达了被压榨的不平感。"},
  {"instruction": "分析这段话的情绪：", "input": "这老板真大方，年终奖发了半年的工资，跟着他干值了。", "output": "【积极赞赏】。对物质激励的正面反馈。"},
  {"instruction": "分析这段话的情绪：", "input": "您这审美真高级，红配绿配紫，真是‘艺术品’。", "output": "【阴阳怪气】。讽刺色彩搭配土气、混乱，表达了嘲弄。"},
  {"instruction": "分析这段话的情绪：", "input": "你这审美真高级，这几个颜色搭在一起很有质感。", "output": "【积极赞赏】。对专业水准的真心赞美。"},
  {"instruction": "分析这段话的情绪：", "input": "这客服态度真好，回复全靠自动回复和复读机。", "output": "【阴阳怪气】。讽刺服务敷衍塞责，表达了解决不了问题的愤怒。"},
  {"instruction": "分析这段话的情绪：", "input": "这客服态度真好，不仅耐心听我讲，还给了额外的补偿方案。", "output": "【积极赞赏】。肯定服务质量，表达了对品牌的信任。"},
  {"instruction": "分析这段话的情绪：", "input": "您这技术可真稳，开个车能把乘客晃晕过去。", "output": "【阴阳怪气】。讽刺驾驶水平差，表达了生理上的不适和嫌弃。"},
  {"instruction": "分析这段话的情绪：", "input": "你这技术可真稳，这山路开得如履平地，厉害！", "output": "【积极赞赏】。认可驾驶技术老练，表达了安全感。"},
  {"instruction": "分析这段话的情绪：", "input": "这空调真给力，开到16度还在出热风。", "output": "【阴阳怪气】。讽刺机器故障，表达了在酷热环境下的烦躁。"},
  {"instruction": "分析这段话的情绪：", "input": "这空调真给力，刚打开屋子里就凉快了。", "output": "【积极赞赏】。对电器性能的直接肯定。"},
  {"instruction": "分析这段话的情绪：", "input": "您可真是懂省钱，给女朋友买礼物都选拼多多九块九包邮。", "output": "【阴阳怪气】。讽刺对方吝啬且不走心，表达了鄙视。"},
  {"instruction": "分析这段话的情绪：", "input": "你可真是懂省钱，这几张优惠券一叠，居然便宜了一百多。", "output": "【积极赞赏】。称赞生活有方，表达了佩服。"},
  {"instruction": "分析这段话的情绪：", "input": "这房子采光真‘好’，白天进屋都得打手电筒。", "output": "【阴阳怪气】。反讽光线极差，表达了对居住环境的嫌弃。"},
  {"instruction": "分析这段话的情绪：", "input": "这房子采光真好，阳光洒进来整个人都暖洋洋的。", "output": "【积极赞赏】。对舒适环境的真诚喜爱。"},
  {"instruction": "分析这段话的情绪: ","input": "你这人真讲‘信用’，说好十点见，现在都快吃午饭了。", "output": "【阴阳怪气】。讽刺对方迟到成性、言而无信。"},
  {"instruction": "分析这段话的情绪: ","input": "你这人真讲信用，答应的事情从来不拖延，靠谱！", "output": "【积极赞赏】。认可人品正直，表达了信赖。"}
]
```

**1\.4 注册数据** 在 `LLaMA-Factory/data/dataset_info.json` 中添加：

```JSON
"sarcasm_data": {
  "file_name": "sarcasm_data.json"
}
```

*\(并把生成的 **`sarcasm_data.json`** 放入 **`data`** 文件夹\)*



## 步骤二：模型微调 \(Training\)

**2\.1 LLaMA\-Factory WebUI 设置**

- **模型：** Qwen2\.5\-3B\-Instruct 。

- **微调方法：** `LoRA`。

- **数据集：** 选择 *`sarcasm_data`*。

- **学习率 \(Learning Rate\)：** `1e-4` \(分类任务学习率可以稍大\)。

- **Epochs：** `10` \(数据少，多学几遍\)。

- **模板 \(Template\)：** `qwen`。

**2\.2 点击“开始训练”** 观察 Loss 曲线。理想情况是 Loss 从 2\.x 降到 0\.1 左右。



**模板 **简单来说，如果模型是“大脑”，对话模板就是“沟通协议”。

### 为什么这个选项至关重要？

不同的模型在预训练或微调时，使用的特殊标记（Special Tokens）是不一样的。

- **Qwen（通义千问）** 使用的是类似 `<|im_start|>user\n...<|im_end|>` 的格式。

- **Llama 3** 使用的是 `<|begin_of_text|><|start_header_id|>user<|end_header_id|>` 的格式。

**如果你选错了：**

- **模型会变“傻”**：它识别不出哪里是用户的指令，哪里是它该回答的地方。

- **复读机现象**：它可能会把你的问题重复一遍，或者输出一堆乱码标记。

- **微调失败**：你刚才准备的那份高质量数据集，如果套用了错误的模板，模型学到的逻辑就会是乱的。

### 对你微调数据的影响

你刚才展示的数据集格式是： `{"instruction": "...", "input": "...", "output": "..."}`

当你选择 **qwen** 模板时，系统会自动把你的数据转换成 Qwen 专属的对话格式：

> `<|im_start|>system 你是一个分析情绪的助手。<|im_end|> <|im_start|>user 分析这段话的情绪：这店家真是天才，把旧货当新品卖。<|im_end|> <|im_start|>assistant 【阴阳怪气】...<|im_end|>`
> 
> 

如果你实际用的是 **Qwen 系列模型**（比如 Qwen2\-7B），这里**必须选 qwen**。

### 如何选择？

- **原则：你底座模型是谁，你就选谁。**

- 如果你用的是 **Qwen/Qwen1\.5/Qwen2**，选 `qwen`。

- 如果你用的是 **Llama 3**，选 `llama3`。

- 如果你用的是 **Yi** 或者 **Mistral**，则需要对应选择它们的模板。

---

## 步骤三：效果验收 \(Inference\)

在 WebUI 的 **Chat** 页面加载模型。

- **测试 Case 1 \(普通差评\)：**

    - 输入：`分析情绪：快递太慢了，等了一周。`

    - *微调前：* 可能会回复“亲，很抱歉给您带来不便\.\.\.” \(像个客服\)

    - *微调后：* **“差评”** \(像个判官，这就对了！\)

- **测试 Case 2 \(阴阳怪气\)：**

    - 输入：`分析情绪: 掌柜的人真好，发过来的鞋子全是灰，帮我省了买旧鞋的钱。`

    - *微调后：* **“阴阳怪气”**



训练之后：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YWNhNjFkNTc0MWIwNGE0Yzk2NzMyNGNiNmUwYmJjNzNfYWNmMWY2OTA4MjJhNDVmY2ZkYWFiZjczODQzNDMyZjNfSUQ6NzYwNDQ4Mjc4MjU4NDIyODgyOF8xNzgxNDQ4NDQ3OjE3ODE1MzQ4NDdfVjM)



![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=M2IwZTU1MWU4MWM2ODlkNTZhYzNiYzgyYzg0ZGE1ZDVfYzZkOTVlYTdjMGJiNmZiMDc5OTc0OTA2MDc5ZmZhOWVfSUQ6NzYwNDQ4NDI1MzM5NjYwMTc5N18xNzgxNDQ4NDQ3OjE3ODE1MzQ4NDdfVjM)



训练之前：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NzljOGNlNDliMzc0ODBhMjNkMGZlOTdkZDEzYzJlY2JfY2NkNGVkMmRmZDkxMTkwYzMxYjk0NmU0MWQwZGEwNWRfSUQ6NzYwNDQ4NDI3Nzc3Mzg4MDUzOV8xNzgxNDQ4NDQ3OjE3ODE1MzQ4NDdfVjM)

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=M2Q3MWQ4OTc1ZDYyMmRlYTk3YTJiMzI2MzU4MmU4YmRfNTVhZTRlZDFmMjdkNWQzMjcwYThhN2EwMDhiZTZjMzZfSUQ6NzYwNDQ4MzI4NDc0Mjk4Mjg2M18xNzgxNDQ4NDQ3OjE3ODE1MzQ4NDdfVjM)





## 真实的企业场景需要考虑哪些问题？

### 训练的数据量选择



模型微调数据、业务非常重要

数据来源：

1. 数据标注工程师标注的数据（公司本身有数据）

    1. 电商公司

    2. 运营提供原始数据（只要评论   运营搞个excel ）

    3. 数据标注工程师，对原始的数据进行标注

    4. 标注完输出  json格式

    5. 写python脚本（Ai写的）转成**Alpaca 格式**

2. Ai生成

    1. 告诉Ai  我的模型训练的目标是啥？

    2. 直接生成**Alpaca 格式**

    3. 你再审核一下



数据量多少合适?

模型微调的  数据质量 \> 数量

1. 先把数据丢给Ai，问 我的质量OK不？

2. 实际需要500\-1000条的数据

3. 数据样本需要多样性，比如说阴阳怪气，可以有很多种表达方式，我们就比较单一。

    1. 假如真实的项目中，我没有这么多数据怎么办？

        1. 先把你现有的数据丢给Ai

        2. 告诉Ai： 我想训练一个模型，需要它能看懂用户的评价，是好评还是差评，还是阴阳怪气

        3. 继续问Ai： 请你看我现在的数据样本够不够？有没有需要补充的

        4. 让Ai继续给你补充数据

        5. 补充到满意为止

    2. 假如真实的项目中，公司已经有数据了

        1. 公司已经有1\-200条真实数据

        2. 把数据丢给Ai

            1. 我的训练目标？

            2. 请看下我的数据样本够不够，需要怎么优化

            3. 让Ai帮你改或者继续加数据



强调，模型训练数据非常关键，开始之前一定要让Ai审核你的数据，Ai最懂Ai，不要凭你自己的感觉



我们目前数据样本的问题：

**模式单一：** 所有的“阴阳怪气”都是通过“褒词贬用”实现的。现实中的阴阳怪气还有：**冷嘲热讽、过度客气、答非所问、甚至只发一个“呵呵”**。

**改进建议：** 1\.  增加一些**中性情绪**的数据（既不是夸奖，也不是阴阳怪气，就是普通描述），防止模型“过拟合”，变得看谁都像在阴阳怪气。 2\.  增加一些**没有关键词**的讽刺，比如：“你这代码写得，连 AI 看了都得沉默三分钟。”（这里没有“真棒”、“天才”这种词）。



### 学习率如何调整？



学习率的调整方向

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MjY2YzcyYTU0MGFmNWRjODJkN2FiNTQ5YzdlMTZkMGZfMjMxOGQyYWM2OGUzZWJiYWZkZTc0NGJlMjViOTJiZWNfSUQ6NzYwNDM4MjgzNjg1ODIyNzk0MF8xNzgxNDQ4NDQ3OjE3ODE1MzQ4NDdfVjM)

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NTlmNGMwY2E0MTg5NDdhNjdmYzI3ODQ3OWM1OWY1ODFfNDU0ZGFhMGM1YmM2ODgwNGQ3MDc5ZjY2ZTA2ZWUzN2FfSUQ6NzYwNDM4MzMyMzc5MTc1NjIzM18xNzgxNDQ4NDQ3OjE3ODE1MzQ4NDdfVjM)



首先：

1. 把业务场景和数据量告诉Ai

2. 让Ai给建议

训练一轮之后，看效果，再调整。



## 训练轮数参数定义

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OWRiZGE0MDE0NzNkOGU1ZGM4ODAwNTk3M2ZmODdlYjhfMDcwZTc4MzczYzdlZDEwNDlhMmY2ZGQyZDkwYjA2NDJfSUQ6NzYwNDM4NDQ5NzU2MzU4NTQ2Nl8xNzgxNDQ4NDQ3OjE3ODE1MzQ4NDdfVjM)

#### **A\. 指令遵循与格式微调（如你现在的“阴阳怪气”识别）**

- **需求：** 让模型学会一种全新的判断逻辑或输出格式。

- **策略：** **多轮数（8\-15 轮）**。

- **原因：** 你需要强行扭转模型原本的语感，让它理解“天才”在特定语境下是坏话。这需要高频次的参数刺激。

#### **B\. 专业知识注入（如“懂王Ai”内部课程知识库）**

- **需求：** 让模型记住新的事实、专业名词或业务流程。

- **策略：** **中等轮数（3\-5 轮）**。

- **原因：** 知识注入不需要改变模型的性格，只需要它“知道”这些内容。轮数过多会导致模型在回答非专业问题时也带入这些专业偏见。

#### **C\. 风格/语气模仿（如抖音商业博主文案分身）**

- **需求：** 模仿某种特定的说话节奏和用词习惯。

- **策略：** **低轮数（2\-4 轮）**。

- **原因：** 风格是非常微妙的，过高的 Epochs 会导致模型只会复读那几句口头禅，丧失基本的逻辑沟通能力。



### 模型参数选择？

涌现。70亿参数是模型能力的分界线。

RTX 5060 8G  = 3b \+ QLORA

而且还是勉强。



真正的模型训练场景

1. 自建机房 一台服务器30w

2. 租算力

3. 云服务   阿里云  火山引擎 



## 训练完成之后

模型微调其实就是不断调节参数的过程。



loss最好能越低越好，降低到1\.0以下。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZTA0NWIzYzQ0ODViM2U5MzNjM2IxMWU3MDc0ZWM5YWNfNWE3ZmY4YTZiNTU2ZDhmNjAzZWM2ZDUzNjE2NTUzMDhfSUQ6NzYwNDQwODE2NjEyMzE3ODk0NV8xNzgxNDQ4NDQ3OjE3ODE1MzQ4NDdfVjM)

### 核心训练指标

- **epoch = 10\.0**

    - **含义：** 训练轮数。意味着你的训练数据集被模型完整地“看”了 10 遍。

    - **解读：** 通常轮数越多，模型对特定数据记忆越深，但也更易产生“过拟合”。

- **train\_loss = 2\.1524**

    - **含义：** 训练损失值。代表模型预测结果与真实标签之间的差距。

    - **解读：** 这个数值本身没有绝对的“好坏”，关键看它相比初始状态是否下降了。2\.15 对于某些微调任务来说是一个比较常见的中途或结尾数值，但最终效果还需结合 `eval_loss` 或实际对话测试。

- **num\_input\_tokens\_seen = 36976**

    - **含义：** 训练过程中模型总共处理的 Token（字符单位）数量。

    - **解读：** 这是计算资源消耗的直接体现。

### 算力与效率指标

- **total\_flos = 576414GF**

    - **含义：** 总浮点运算次数（FLOPs），G 代表 10 亿。

    - **解读：** 这是衡量硬件出了多少“力”的硬指标。数值越大，说明模型越大或计算任务越重。

- **train\_runtime = 0:01:15\.05**

    - **含义：** 训练总耗时。

    - **解读：** 你的训练非常快，只用了 **1 分 15 秒**。这说明你的数据集可能比较小，或者显卡性能非常强。

- **train\_samples\_per\_second = 6\.662  （大模型每秒处理多少数据，你可以计算我一次训练多久）**

    - **含义：** 每秒处理的样本数。

    - **解读：** 衡量训练吞吐量。每秒能跑 6\.6 个样本，效率尚可。

- **train\_steps\_per\_second = 0\.533    （步频： 我一次喂给大模型多少数据去学习，小了太局限，大了又太囫囵吞枣）**

    - **含义：** 每秒完成的训练步数（Step）。

    - **解读：** 一个 Step 通常包含一个 Batch 的数据。这里大约 2 秒完成一个 Step。



总结：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YzU2NDgxMzViZjQwOWU0MWQyMjlkZjY0NjMyM2UxMTJfNjFhYjA0ODNhMTgwN2Q4MDEyOTljZjI0MzMzNjFhZTNfSUQ6NzYwNDQ5MjU1NjM3MzYyNjA1NV8xNzgxNDQ4NDQ3OjE3ODE1MzQ4NDdfVjM)

## Qwen2\.5\-3B和Qwen2\.5\-3B\-Instruct 模型的区别？

### Qwen2\.5\-3B \(Base 模型\)

这是**预训练模型（Base Model）**。它在数万亿 Token 的文本上进行了学习，本质是一个极其强大的\*\*“文本补全器”\*\*。

- **特点**：它不知道什么是“指令”，也不习惯对话。如果你问它“请帮我写一首诗”，它可能不会写诗，而是接着你的话补充：“\.\.\.并评价这首诗的修辞手法。”（因为它认为你在给它一段作业题目）。

- **用途**：

    - 作为**微调的起点**。如果你有大量的特定领域数据（比如医疗、法律），你应该用 Base 模型开始训练。

    - 用于**续写**故事或完成固定的文本格式。

### Qwen2\.5\-3B\-Instruct \(指令微调模型\)

这是在 Base 模型基础上，经过了 **SFT（监督微调）** 和 **RLHF（人类反馈强化学习）** 后的版本。

- **特点**：它被教会了**如何听懂人类的指令**。它知道什么是对话，知道要礼貌地回答问题，也知道拒绝不当请求。

- **用途**：

    - **直接使用**。聊天机器人、写代码、总结摘要、日常助手。

    - **少量样本微调**。就像你现在只有 50 条数据的情况，通常建议基于 **Instruct** 版本做微调，因为模型已经具备了基本的对话逻辑，你只需要给它“喂”一点点特定领域的知识即可。





## 第二次训练的结果



loss的结果是1\.3



典型： 学会格式，丢了脑子。过拟合。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YWY4OWI3MzBhN2NmODEzOTJkZjRmMzQwYTNiYjNjNjdfZjM5ZTExMWUzZTIzYjRmOWZhYjQwN2U4OWVjNjllMjFfSUQ6NzYwNDQ4NTQyODAwNDY3MDY3MF8xNzgxNDQ4NDQ3OjE3ODE1MzQ4NDdfVjM)

现在的参数

2e\-4

轮数 10轮



三个问题：

1. 学习率太高了，影响了模型本身的逻辑

2. 轮数过多，过拟合了。模型只会死记硬背

3. 逻辑不行  提升rank





第三轮训练（降的太慢）

5e\-5

rank16

3轮

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MDIyMTg1NjZhZDkwMTE2NWZiY2VhYjcwODdhNWY3MTNfMTlkY2ExZDk0ODdiYTMwOWI3NTE3ZGJhYjYxNzFiMzFfSUQ6NzYwNDQ4OTYxNTUyNDg3NTIwM18xNzgxNDQ4NDQ3OjE3ODE1MzQ4NDdfVjM)



第四轮训练：

结果：

有进步。能理解用户的情绪，但是隐喻的话还是有点困难。





![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MjAyNGMyM2Q5N2Y2Y2E3M2RmYTQ3MWUxYmRjMTQ2NTdfYzY3ODFiZjUxODk3YzFhNWYyNGE5OTY3MjBkYTEzMWFfSUQ6NzYwNDQ5NTUwOTc5NzA4MDAwOF8xNzgxNDQ4NDQ3OjE3ODE1MzQ4NDdfVjM)

1e\-4

8轮

Rank 16





1\.3\(过拟合\) =\> 1\.8



loss越大   越欠拟合

越小，越过拟合

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NDAxNTE0ZWY2MzIwOWRiMTRiYzVjNjUxZmVlODFkYWRfMmZkNjJkMTVmOGEwM2Y3ZDU3Y2Q1N2QxZjExNTY1YjdfSUQ6NzYwNDQ5NDIxNzM4OTM2MjExOV8xNzgxNDQ4NDQ3OjE3ODE1MzQ4NDdfVjM)

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MDkyM2VhMTNjNDQ1YmQyZjY2N2FlYTljZWI5NzMxNDRfMGUxMWFhMDI4YTY0MWFiNGExNGRiNjRkYmYxYjRlYmRfSUQ6NzYwNDQ5NDg3ODU0NjA4Njg1Ml8xNzgxNDQ4NDQ3OjE3ODE1MzQ4NDdfVjM)





继续优化的方向：

### 扩充“反讽”的多样性（数据侧优化）

目前模型对明显的负面词汇（如拉肚子）反应灵敏，但对这种看似“省钱、好人”实则抱怨的逻辑还很模糊。

- **增加语义冲突样本**：专门搜集 5\-10 条这种“看似获利，实则受损”的数据加入训练集。

    - *例：* “老板人真好，过年还让我留在公司看大门，工资都不用发，直接帮我戒掉了消费欲望。” \-\> 【阴阳怪气】。

- **引入“负面拒绝”样本**：在数据中混入一些**完全没有阴阳怪气**的正常差评。

    - *例：* “鞋子全是灰，质量太差了，要求退货。” \-\> 【强烈不满】。

    - **目的**：让模型明白，只有当“好听的词”和“糟糕的事实”同时出现时，才是阴阳怪气。

### 微调参数：增加 LoRA Alpha

你目前的 Rank 是 16，如果模型还是学得不够“深刻”，可以调整一个参数：

- **LoRA Alpha**：通常建议设为 **Rank 的 2 倍**（即设为 32）。

- **原理**：Alpha 决定了微调层对原模型的影响权重。稍微调大一点，能让模型更果断地应用你教给它的新逻辑。

### 尝试使用“思维链” \(CoT\) 引导

你的数据集 `output` 只有结果和简单解释。你可以尝试让模型在输出标签前，先**拆解逻辑**。

- **修改训练数据的 Output 格式**：

> - “这段话表面上在夸奖【掌柜人好】和【省钱】，但实际事实是【鞋子全是灰】，存在明显的逻辑冲突，因此判定为【阴阳怪气】。”
> 
> 

- **效果**：这种“先推理再下结论”的方式能显著提升 3B 小模型的逻辑判断准确率。

### 训练策略：尝试余弦预热 \(Warmup\)

由于你现在的学习率已经很低（1e\-5），可以配合 **Warmup Ratio: 0\.1**。

- **作用**：在 8 轮训练的前 10% 时间里，学习率从 0 慢慢升到 1e\-5。这能防止模型在第一轮就因为某些极端样本把参数“带歪”。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MTk4ZTRkOTk4MDExNjI5MGVlMjE0YzZiMWVkZjZmZmNfYWM1NzQ4ZTlmYTI1NGQzZTA4MWUyNTVmYWU1NGY3MTBfSUQ6NzYwNDQ5NzkzMjcyMDYzOTE3N18xNzgxNDQ4NDQ3OjE3ODE1MzQ4NDdfVjM)



