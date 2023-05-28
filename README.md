# AICongyin-LLM.github.io
AGI LLM 调研资料汇总，这里提供的资料尽可能记录关键词与链接，方便大家进行深入调研和挖掘。

# 5.7 周7 及之前
* 1、ChatGPT 调研报告（仅供内部参考）  哈尔滨工业大学自然语言处理研究所（HIT-NLP）2023年3月6日
* 2、Open Chat Video Editor：开源的短视频生成和编辑工具，Python 在线聊天视频编辑器，能实现快速将聊天记录转换为视频格式，支持自定义主题风格和配乐，并能输出多种视频格式。包含了丰富的 API 和示例代码，方便开发者进行二次开发和集成: github.com/SCUTlihaoyu/open-chat-video-editor
* 3、ChatZoo：对话语言模型横向对比工具。可以将自有模型或者来自 🤗huggingface 的模型轻松部署到网页中。ChatZoo 还可以将一句提示同时发送到多个模型中进行回答生成，方便地对比模型效果： github.com/OpenLMLab/ChatZoo
* 4、VardaGPT：在GPT-2模型基础上添加了一种称为关联记忆(associative memory)的机制。关联记忆可以使模型在处理文本时更好地捕捉上下文信息。: github.com/ixaxaar/VardaGPT
* 5、Toolformer：通过prompt令语言模型生成一些能够调用工具的样本，然后利用语言模型对文本的困惑度来过滤这些样本，得到高质量的数据集。最后，利用高质量数据集微调语言模型，赋予其使用工具的能力。
* 6、ART：人工构建一下常见的工具使用样本，然后利用新任务与已有任务的相似性来辅助语言模型使用工具。
* 7、InferLLM：非常轻量的 LLM 模型推理框架，可以本地部署 LLM 中的量化模型，推理速度还不错。: github.com/MegEngine/InferLLM
* 8、用RedPajama基础数据集训练的RedPajama-INCITE模型推出，有3B和7B两个版本，包括指令微调和聊天的版本：https://www.together.xyz/blog/redpajama-models-v1
* 9、Open LLMs：可供商业使用的开放大型语言模型(LLM)列表github.com/eugeneyan/open-llms
* 10、Auto-evaluator：文档问答的自动评估: github.com/langchain-ai/auto-evaluator
* 11、MosaicML NLP团队发布了一个开源的、可商用的语言模型MPT-7B，效果对标LLaMA-7 B。同时发布的还有基于此模型的三个微调的变体，特别是MPT-7B-StoryWriter-65k+ ：一个设计用来阅读和编写超长上下文长度的故事的模型。上下文长度为65 k：https://www.mosaicml.com/blog/mpt-7b
* 12、Fixing Hallucination with Knowledge Bases  https://www.pinecone.io/learn/langchain-retrieval-augmentation/


# 5.8 周1
* 1、Plan，Eliminate,and Track -Language Models are Good Teachers for Embodied Agents. autogpt系列的另外一个工作
* 2、Donut 🍩 : Document Understanding Transformer
* 3、多模态语言-图像数据集、LLaVA模型及在线Demo：利用语言模型生成多模态语言-图像指令遵循数据，并用这些数据训练出大型多模态模型LLaVA，用于通用的视觉和语言理解。用语言模型GPT-4生成多模态指令遵循数据，并在HuggingFace Dataset上公开了15.8万条样本；将预训练的CLIP ViT-L/14视觉编码器和大型语言模型LLaMA连接起来，并采用了两阶段的指令微调过程：http://aicoco.net/s/2u
* 4、5月6日，ChatGPT母公司OpenAI发布了最新开源项目Shap-E，通过文本就能生成3D模型。目前github已经突破2000颗星。（开源地址：https://github.com/openai/shap-e）

# 5.9 周2
* 1、ICLR 2023 | DIFFormer: 扩散过程启发的Transformer
* 本⽂介绍⼀项近期的研究⼯作，试图建⽴能量约束扩散微分⽅程与神经⽹络架构的联系，从而原创性的提出了物理启发下的 Transformer，称作 DIFFormer。作为⼀种通⽤的可以灵活⾼效的学习样本间隐含依赖关系的编码器架构，DIFFormer 在各类任务上都展现了强大潜⼒。
* 2、SmartGPT https://mp.weixin.qq.com/s/rlu1we8YPJZcu_13iB9obg
能让ChatGPT完成复杂任务，GPT3.5和GPT-4都支持。
它通过将问题拆解，并调用外部资源，提高了GPT的工作能力。
在它的调教下，GPT-4回答的准确率从68%提高到了85%。
* 3、ChatGPT-Developer-Plugins：不需要 Plus 订阅直接运行 ChatGPT 插件的开发者工具。提供了多种插件，包括翻译、摘要、新闻、笑话等。用户可通过简单的命令行操作使用这些插件，并获取其输出。该工具用 Python 编写，免费提供给所有人使用: github.com/SamurAIGPT/ChatGPT-Developer-Plugins
* 4、LLaMA Generative Agent：针对LLaMA模型的生成式智能体实现，派生自langchain实现。旨在实现一种生成文本的机制，可用于生成各种文本，包括但不限于电子邮件、文章和评论等。: github.com/UranusSeven/llama_generative_agent
* 5、Langcorn：用 LangChain + FastApi 部署自动化LLM应用，以便将 LangChain 应用作为 API 提供服务: github.com/msoedov/langcorn
* 6、ProgGP：包含173首前卫金属风格歌曲的数据集，包括 GuitarPro 格式和Token格式，符合 DadaGP 规范。该数据集可以帮助研究人员和音乐家进行音乐分析和生成相关研究: github.com/otnemrasordep/ProgGP
* 7、Few-shot In-context Learning for Knowledge Base Question Answering  KBQA

# 5.10 周3
* 1、【OpenBuddy：一款强大的开源多语言聊天机器人模型，目标是全球用户，重点是对话AI和流畅的多语言支持，包括英文、中文等多种语言。基于Facebook的LLAMA模型，进行了微调，包括扩展词汇表、增加常用字符和增强的token embeddings。通过这些改进和多轮对话数据集，OpenBuddy提供了一个强大的模型，能回答问题并在各种语言之间进行翻译任务。: github.com/OpenBuddy/OpenBuddy
* 2、LLM-Leaderboard：由社区联合创建的用于展示大型语言模型(LLM)的集中式排行榜: github.com/LudwigStumpp/llm-leaderboard
* 3、InfiniteGPT：一个Python脚本，允许您向OpenAI API输入无限大小的文本，从而消除了在使用大量文本输入或将无尽的文本块复制粘贴到chatGPT时重新提示的需求: github.com/emmethalm/infiniteGPT。
* 4、PaLM的开源复现。已训练了三种不同大小的PaLM模型（150m，410m，1b），并且还在训练一个2b模型: github.com/conceptofmind/PaLM
* 5、HugNLP：基于Hugging Face Transformer的统一和全面的自然语言处理(NLP)库，旨在提高NLP研究人员的便利性和有效性。该库提供了如BERT、RoBERTa、GPT-2等流行的transformer-based模型，以及一种名为KP-PLM的知识增强预训练范式。HugNLP还实现了一些针对特定任务的模型，包括序列分类、匹配、标注、span提取、多选择和文本生成。此外，该库还支持少样本学习环境，提供了一个原型网络，用于少样本文本分类和命名实体识别(NER)。: github.com/HugAILab/HugNLP
* 6、DB-GPT：基于vicuna-13b和FastChat的开源实验项目，采用了langchain和llama-index技术进行上下文学习和问答。项目完全本地化部署，保证数据的隐私安全，能直接连接到私有数据库处理私有数据。其功能包括SQL生成、SQL诊断、数据库知识问答等: github.com/csunny/DB-GPT
* 7、SmartGPT：旨在为大型语言模型(尤其是GPT-3.5和GPT-4)提供完成复杂任务的能力，通过将它们分解成更小的问题，并使用互联网和其他外部来源收集信息。特点包括模块化设计，易于配置，以及对插件的高度支持。SmartGPT的运作基于"Autos"的概念，包括"Runner"和"Assistant"两种类型，都配有处理计划、推理和任务执行的LLM代理。此外，SmartGPT还具有内存管理系统，以及可以定义各种命令的插件系统: github.com/Cormanz/smartgpt

# 5.11 周4
* 1、ImageBind，模型可以横跨 6 种不同的模态（图像、文本、音频、深度、温度和 IMU 数据）进行联动交流：https://github.com/facebookresearch/ImageBind \
论文地址：dl.fbaipublicfiles.com/imagebind/imagebind_final.pdf \
项目地址：github.com/facebookresearch/ImageBind \

# 5.12 周5
* 1、CAMEL: 从LLaMA衍生并适应临床的模型。CAMEL基于LLaMA进行进一步的微调，使用了MIMIC-III和MIMIC-IV的临床病例，并在临床指导上进行微调: github.com/starmpcc/CAMEL
* 2、Shap-E: OpenAI发布的全新隐式text-to-3D模型，速度快，但是生成性能略有不足。论文链接：https://arxiv.org/pdf/2305.02463.pdf 代码链接：https://github.com/openai/shap-e
* 3、《语言模型可以解释语言模型中的神经元》（Language models can explain neurons in language models），OpenAI在官网发布了的博文，用GPT-4解释GPT-2的行为。论文地址：https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html#sec-intro
* 4、chinese-StableVicuna，全球首个StableVicuna中文优化版，全球首发最强开源中文GPT模型。StableVicuna基于Vicuna-13B模型实现，是全球首个基于--RLHF人类反馈训练--的开源LLM模型，被业界视为：是自ChatGPT推出以来的第二个里程碑。代码链接：https://github.com/ziwang-com/chinese-StableVicuna.git
* 5、BMTools BMTools是一个开源存储库，使用工具扩展语言模型，并作为社区构建和共享工具的平台。在这个存储库中，您可以（1）通过编写python函数轻松构建插件（2）使用外部ChatGPT插件。该项目的灵感来自开源项目LangChain，并针对ChatGPT插件等开源工具的使用进行了优化，努力实现ChatGPT Plugins的开源学术版。代码链接：https://github.com/OpenBMB/BMTools

# 5.13 周6
* 1、starcoder：BigCode社区推出了StarCoder和StarCoderBase，它们是具有155亿参数和8K上下文长度的模型，具备填充功能，并且通过多查询注意力实现了快速的大批量推理。论文链接：https://arxiv.org/pdf/2305.06161.pdf
* 2、A Survey of Large Language Models，大语言模型综述，论文链接：https://arxiv.org/pdf/2303.18223.pdf

# 5.14 周7
* 1、Introduction to Generative AI，【生成式人工智能入门】，视频链接：https://www.youtube.com/watch?v=G2fqAlgmoPo
* 2、大语言模型（LLM）微调技术笔记。https://github.com/ninehills/ninehills.github.io/issues/92
* 3、tokenmonster能在给定文本数据集、词汇量大小和最大词汇长度的情况下，选择在该词汇量大小下最优化表示数据集的词汇。:github.com/alasdairforsythe/tokenmonster
* 4、datasetgan 是如何生成有标签的数据集的？DatasetGAN 是 UC Berkeley 计算机科学与人工智能实验室 (BAIR) 提出的一种算法，用于生成有标签的数据集，基于生成对抗网络 (GAN) 技术，可以生成高质量的有标签样本，以解决现有数据集较小或有限的问题。 具体来说，DatasetGAN 的生成过程包括以下步骤： 1. 利用无标签的原始数据集训练 GAN 模型。该步骤旨在学习原始数据集的生成分布。GAN 模型由生成器和判别器两部分组成。生成器被训练为生成与原始数据集分布相似的数据。判别器被训练为区分原始数据集和生成数据集之间的区别。 2. 利用生成器生成一部分伪造数据集。生成器通过学习原始数据集的分布，尝试生成具有类似特征的伪造数据集。 3. 利用人工标注对伪造数据集进行分类标记。这是指利用数据集中某些数据的真实标签信息，手动标注伪造数据集，以使其具有分类标记。 4. 利用有标签样本和无标签伪造数据去微调训练模型。除了原始的有标签样本之外，还有一个包含有标签和无标签数据的数学集。有标签的数据用于模型的精细分类，无标签的伪造数据则被用来增加数据集和模型的泛化能力。 5. 利用微调后的模型，对伪造数据集进行再次筛选，去除一些错误标签的样本，以获得更高质量的有标签数据集。 通过DatasetGAN生成的有标签数据集，可以提高深度学习和人工智能模型的溯源过程，同时还可以促进更广泛的科学门类、域和任务上的机器学习研究和应用。官网链接：https://nv-tlabs.github.io/datasetGAN/ 论文链接：https://arxiv.org/pdf/2104.06490.pdf 代码链接：https://github.com/nv-tlabs/datasetGAN_release
* 5、ChatGPT插件开发。笔记链接：https://techdiylife.github.io/ChatGPT-programming-handbook/contents/chat-plugins-overview.html

# 5.15 周1
* 1、【2023新书】生成深度学习：教机器绘画、写作、作曲和游戏 第二版，453页pdf 资料链接：https://mp.weixin.qq.com/s/GXqUFKGp0ySmTfTzpwsFMw
* 2、Chat2Plot：交互式且安全的文本到可视化库，使用LLM来生成可视化图表、高级图表规格(以json格式: github.com/nyanp/chat2plot

# 5.16 周2
* 1、Prompt Sapper：基础模型的灵魂伴侣，AI服务的创新工场\
澳大利亚 Data61 的 SE4AI 团队和江西师范大学智能化软件工程实验室联合打造全球首款 AI 链（AI chain）无代码生产平台 Prompt Sapper，及相应的方法学和 AI 服务市场。基础模型（foundation models）带来了前所未有的 AI “操作系统” 效应和全新的人工智能交互方式，激发了 AI 服务开发与应用的创新之潮。
项目链接: https://github.com/AI4FutureSE  \
AI 链主网站: https://www.aichain.online/ \
Sapper IDE:  https://www.promptsapper.tech/ \
AI 服务市场：https://www.aichain.store/ \

# 5.17 周3
* 1、Salesforce发布新模型InstructBLIP，基于BLIP2使用指令微调 \
华人团队开源指令精调的InstructBLIP多模态大模型，横扫多项SOTA，看图&推理&问答&对话样样通！\
CVHub文章链接：https://mp.weixin.qq.com/s/XezIEmnFzVWbt4ZUt-3dmw \
论文地址：arxiv.org/abs/2305.06500 \
代码地址：github.com/salesforce/LAVIS/tree/main/projects/instructblip
* 2、Plan-and-Solve Prompting，一个新的Prompt，可以帮助大语言模型求解复杂的问题。之前，用Let's think step by step可以触发LLM产生链式思考，提升解决问题的准确度，现在用这个新的办法，可以进一步提升LLM准确度。\
论文地址：arxiv.org/abs/2305.04091
* 3、外行也能看懂的大语言模型结构对比！\
https://mp.weixin.qq.com/s/Ja6eUSpzQuifJld1QVIfOA
* 4、DeepMind 近日发布了一个新型数据集，包含大量不同类型的数学问题（练习题级别），旨在考察模型的数学学习和代数推理能力。\
数据集地址：https://github.com/deepmind/mathematics_dataset \
目前该数据集发布了 1.0 版，其每个模块包含 200 万（问题答案）
* 5、大语言模型数据集整理  https://opendatalab.org.cn
* 6、因果推理与大语言模型：开辟因果关系的新前沿 \
集智俱乐部文章链接：https://mp.weixin.qq.com/s/nvREvjYW1dEcqDrTo6b9zQ \
论文题目：Causal Reasoning and Large Language Models: Opening a New Frontier for Causality \
论文链接：https://arxiv.org/abs/2305.00050 \
文章题目：On the unreasonable effectiveness of LLMs for causal inference \
文章链接：https://threadreaderapp.com/thread/1653457971844874240.html 
* 7、SuperICL: In this paper, we propose Super In-Context Learning (SuperICL) which allows black-box LLMs to work with locally fine-tuned smaller models, resulting in superior performance on supervised tasks. Our experiments demonstrate that SuperICL can improve performance beyond state-of-the-art fine-tuned models while ad- dressing the instability problem of in-context learning. Furthermore, SuperICL can enhance the capabilities of smaller models, such as multilinguality and interpretability. \
paper url:https://arxiv.org/pdf/2305.08848v1.pdf \
code url:https://github.com/JetRunner/SuperICL

# 5.18 周4
* 1、Ahead of AI: The Latest Open Source LLMs and Datasets \
https://magazine.sebastianraschka.com/p/ahead-of-ai-8-the-latest-open-source
* 2、大模型如何端边部署？华盛顿大学Google提出《逐步蒸馏》法，以更少的训练数据和更小的模型规模超越更大的语言模型 \
专知文章链接：https://mp.weixin.qq.com/s/yKyGJczqfvXsT0GXhcfuPQ \
论文链接：https://www.zhuanzhi.ai/paper/fa04cb640eb5b7dd65cddc946c76b80f
* 3、图文理解能力强大！多模态对话生成模型：mPLUG-Owl，已开源！\
文章链接：https://mp.weixin.qq.com/s/tQYV54g6aMJxogmI3MzmiA \
论文链接：https://arxiv.org/abs/2304.14178 \
项目链接：https://github.com/X-PLUG/mPLUG-Owl \
在线demo：https://modelscope.cn/studios/damo/mPLUG-Owl/summary \
达摩院猫头鹰mPLUG-Owl亮相：模块化多模态大模型，追赶GPT-4多模态能 https://mp.weixin.qq.com/s/otq9tpmGu3UXEWOYSHTvnA
* 4、中科院发布多模态 ChatGPT，图片、语言、视频都可以 Chat ？中文多模态大模型力作 \
文章链接：https://mp.weixin.qq.com/s/RqiJvhH4sdtHBVIDZXmu5Q
论文题目：X-LLM: Bootstrapping Advanced Large Language Models by Treating Multi-Modalities as Foreign Languages \
论文链接：https://arxiv.org/pdf/2305.04160.pdf \
项目主页：https://x-llm.github.io/ 
* 5、Transformers Agent，全球最火的AI社区HuggingFace官方出品「Transformers Agent」，通过控制10万多个AI，也能实现魔法。在HuggingGPT的基础上的进一步的工作 \
文章链接：https://mp.weixin.qq.com/s/YY0ecZAsytEDUXMyj9NK4g
参考资料：\
https://twitter.com/huggingface/status/1656334778407297027 \
https://huggingface.co/docs/transformers/transformers_agents
* 6、【CMU博士论文】弥合人类视觉与计算机视觉之间的鸿沟,144页pdf
文章链接：https://mp.weixin.qq.com/s/KqAJds7F7C4fxgmJxSh7MA
* 7、OpenAI用GPT-4解释了GPT-2三十万个神经元 \
相关文章链接：\
https://mp.weixin.qq.com/s/aBbDW_PA-5ze5WMg1lsgQQ \
https://mp.weixin.qq.com/s/pKP1pRenwcQMByFJyiuRKQ
* 8、能看图、会聊天，还会跨模态推理和定位，能落地复杂场景的DetGPT来了，港科大LMFlow团队 & 港大NLP实验室 \
文章链接：https://mp.weixin.qq.com/s/elVSpp_pQKp_XCgWLmhwRQ \
开源代码：https://github.com/OptimalScale/DetGPT \
Demo 在线试玩：https://detgpt.github.io/ 
* 9、MathGPT来了！专攻数学大模型，解题讲题两手抓，学而思自研数学大模型MathGPT \
文章链接：https://mp.weixin.qq.com/s/RUnJ2T9BueDnDCu91m8uPQ \
* 10、大模型“涌现能力” ？！斯坦福最新研究：别迷信，这是度量选择的结果 \
文章链接：https://mp.weixin.qq.com/s/9LMB5HXmXU3dbqaMZfmG1Q \
论文：https://arxiv.org/pdf/2304.15004.pdf
* 11、ACT-1: Transformer for Actions 一个通用的、能帮助用户在各种软件上完成任务的AI助手 \
文章链接：https://zhuanlan.zhihu.com/p/565025337 \
官网：https://www.adept.ai/blog/act-1 
* 12、GPT充当大脑，指挥多个模型协作完成各类任务，通用系统AutoML-GPT来了 \
AutoML-GPT 使用 GPT 作为各种 AI 模型之间的桥梁，并用优化过的超参数来动态训练模型。AutoML-GPT 动态地接收来自 Model Card [Mitchell et al., 2019] 和 Data Card [Gebru et al., 2021] 的用户请求，并组成相应的 prompt 段落。最后，AutoML-GPT 借助该 prompt 段落自动进行多项实验，包括处理数据、构建模型架构、调整超参数和预测训练日志。\
AutoML-GPT 通过最大限度地利用其强大的 NLP 能力和现有的人工智能模型，解决了各种测试和数据集中复杂的 AI 任务。大量实验和消融研究表明，AutoML-GPT 对许多人工智能任务（包括 CV 任务、NLP 任务）是通用的、有效的。\
论文地址：https://papers.labml.ai/paper/35151be0eb2011edb95839eec3084ddd \
文章链接：https://mp.weixin.qq.com/s/DGrWcoJv2AQXiL_bNM8z0Q \
13、6G显存玩转130亿参数大模型，仅需13行命令，RTX2060用户发来贺电\
文章链接：https://mp.weixin.qq.com/s/hvRVSwlhWKcZFAtl617F5A \
14、

# 5.19 周5
* 1、GPT BAT：GPT长文本批处理工具，可以将长文本分隔成小段，然后使用GPT进行处理，并将结果拼接起来以便下载。使用该工具需要选择分隔方式(按行、按长度或按特殊字符)，填写每次调用GPT Chat API的设置，包括系统提示词、用户提示词、最大Token数和模型。: github.com/easychen/gpt-bat
* 2、Chinese-CLIP \
地址：github.com/OFA-Sys/Chinese-CLIP \
本项目为CLIP模型的中文版本，使用大规模中文数据进行训练（~2亿图文对），旨在帮助用户快速实现中文领域的图文特征&相似度计算、跨模态检索、零样本图片分类等任务。
* 3、目前的多模态大语言模型多采用外接一个其它模态的编码器。但是这离AGI还有一定的距离，我们提出了SpeechGPT，它具有内生的跨模态能力，是第一个既能接受跨模态输入，也能产生跨模态输出的大语言模型。SpeechGPT突破了传统语音到语音对话cascaded system (ASR+LLM+TTS) 的束缚，实现了模态之间的知识传递，不需要额外的ASR和TTS系统也能和LLM直接进行语音对话。 我们利用语音离散表示来统一了语音和文本的符号表示，通过扩充LLM词表的方式自然地把语音模态集成到LLM之中。并且构造了第一个语音-文本跨模态指令微调数据集SpeechInstruct，经过modality-adaptation pre- training, cross-modal instruction fine-tuning, chain-of-modality instruction fine-tuning三阶段的训练，使得模型具有不错的跨模态指令遵循能力和语音到语音对话的能力。在我Demo page里，我们展示了SpeechGPT可以充当会说话的百科全书，生活助手，闲聊伙伴，诗人，心理医生，学习助手等等。。。 SpeechGPT为打造真正的多模态大语言模型指明了方向：将不同模态的数据（视觉，语音等）统一表示为离散单元集成在LLM之中，在跨模态数据集上经过预训练和指令微调，来使得模型具有多模态理解和生成的能力，从而离AGI更进一步。 \
Demo page: https://0nutation.github.io/SpeechGPT.github.io/ \
Paper: https://arxiv.org/abs/2305.11000 \
Github: https://github.com/0nutation/SpeechGPT/tree/main \
为多模态LLM指明方向，邱锡鹏团队提出具有内生跨模态能力的SpeechGPT https://mp.weixin.qq.com/s/KpdOUdeYSVzrBtfuqFbjaQ
* 4、DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining \
* 5、谷歌PaLM 2细节曝光：3.6万亿token，3400亿参数 文章链接：https://mp.weixin.qq.com/s/xAhbvd1zo4v91XX3Mrh9jA
* 6、解释大语言模型：在 Alpaca 中识别因果机制 \
斯坦福大学的 Alpaca 模型是用于学术研究的指令遵循语言模型（instruction-following language model）。在近日新发表的论文“解释大语言模型：在 Alpaca 中识别因果机制”中，研究者提出一种通用的因果机制发现框架，使用该工具，Alpaca 模型在简单的数字推理任务中实现了具有可解释中间变量的因果模型。这些因果模型对于输入和指令的变化具有鲁棒性。该框架也适用于拥有数十亿参数的大语言模型。
论文题目：Interpretability at Scale: Identifying Causal Mechanisms in Alpaca\
论文链接：https://arxiv.org/abs/2305.08809\
作者：Zhengxuan Wu（吴政璇）, Atticus Geiger, Christopher Potts, Noah Goodman 文章链接：https://mp.weixin.qq.com/s/dWOD7G6is0pvKFbLbWQhfg

# 5.20 周6
* 1、PyLLMs：一个简洁的 Python 库，用于连接各种 LLM(OpenAI、Anthropic、Google、AI21、Cohere、Aleph Alpha、HuggingfaceHub)，内置模型性能基准。非常适合快速原型设计和评估不同模型，具有以下特点：通过少量代码连接顶级 LLM；响应元数据包括处理的Token、成本和延迟，对各个模型进行标准化；支持多模型：同时从不同模型获取补全；LLM 基准：评估模型的质量、速度和成本: github.com/kagisearch/pyllms
* 2、Redis-LLM-Document-Chat：用LlamaIndex、Redis和OpenAI与PDF文档进行交互，包含一个Jupyter笔记本，演示了如何使用Redis作为向量数据库来存储和检索文档向量，还展示了如何使用LlamaIndex在文档中执行语义搜索: github.com/RedisVentures/LLM-Document-Chat
* 3、Scikit-LLM: 用于增强文本分析任务的工具，可以无缝地将ChatGPT等强大的语言模型集成到Scikit-Learn中。提供了ZeroShotGPTClassifier类和MultiLabelZeroShotGPTClassifier类，用于进行零样本文本分类和多标签零样本文本分类。: github.com/iryna-kondr/scikit-llm
* 4、Transformer 估算 本文主要介绍用于估算 transformer 类模型计算量需求和内存需求的相关数学方法。\
https://mp.weixin.qq.com/s/j8vw9sAdG_Vfh-i0SN9p6Q
* 5、非常详细！大型自然语言模型（LLM）发展概要，及其关键技术！\
https://mp.weixin.qq.com/s/RTJndbXsVewHUNtj6T189Q
* 6、大模型阅读笔记：ChatGLM-6B模型结构组件源码阅读 \
https://mp.weixin.qq.com/s/bEBrWooUU2MuMt8DwzsDKQ
* 7、ICLR2023高分论文 | ACT:2D视觉或语言 Foundation Model可以帮助3D表征学习吗? https://mp.weixin.qq.com/s/y2GcwqHug5EB2jlwnlxKTA
Title: Autoencoders as Cross-Modal Teachers: Can Pretrained 2D Image Transformers Help 3D Representation Learning?
Paper: https://arxiv.org/abs/2212.08320
Code: https://github.com/RunpeiDong/ACT

# 5.21 周7
* 1、5月20日，马克斯普朗克研究、麻省理工计算机与AI实验室、via-center、宾夕法尼亚大学和谷歌等联合发布了一篇名为《Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold》的论文。根据其Github主页显示，预计6月进行开源。该论文提出了一种控制GAN的新方法 DragGAN，使用户能以无损的方式拖动几下鼠标就能改变图像形态。例如，将一张严肃的脸，拖动几下鼠标就能变成微笑的形态；将一匹站立的马，更改成奔跑形态等。Github地址：https://github.com/XingangPan/DragGAN项目展示：https://vcai.mpi-inf.mpg.de/projects/DragGAN/论文地址：https://arxiv.org/abs/2305.10973

# 5.22 周1
* 1、Plug and Plai：简化将AI插件集成到开源语言模型(LLMs)的开源库，提供实用函数来从plugnplai.com目录获取插件列表，获取插件清单，提取OpenAPI规范并加载插件: github.com/edreisMD/plugnplai
* 2、吴恩达 x OpenAI Prompt Engineering教程中文笔记  https://mp.weixin.qq.com/s/LNPm5dk9pqN7dsx6MFDbTA、
* 3、强过AutoGPT！微软重磅研究提出APO算法，「自动提示」淘汰提示工程师 https://mp.weixin.qq.com/s/Ryy7Yg2S3gCp11g7HMBuWw \
《Automatic Prompt Optimization with "Gradient Descent" and Beam Search》 \
论文地址：https://arxiv.org/pdf/2305.03495.pdf
* 4、Meta最新模型：LIMA-65B，没有RLHF，模型效果远胜Alpaca！！ https://mp.weixin.qq.com/s/cA6HoPsLhPdQ_ntlL2MKDw \
《LIMA:Less Is More for Alignment》 \
论文：https://arxiv.org/pdf/2305.11206.pdf
* 5、有证据了，MIT表明：大型语言模型≠随机鹦鹉，确实能学到语义 https://mp.weixin.qq.com/s/sx5FsFZBrXaWOqbtJUPvIQ \
《Evidence of Meaning in Language Models Trained on Programs》 \
论文地址：https://paperswithcode.com/paper/evidence-of-meaning-in-language-models

# 5.23 周2
* 1、大模型与联邦学习 《Towards Building the Federated GPT: Federated Instruction Tuning》论文
* 2、思维树ToT: GPT-4推理提升1750%！普林斯顿清华姚班校友提出全新「思维树ToT」框架，让LLM反复思考 https://mp.weixin.qq.com/s/1SswD6i6lGxKAvU-pzz-6A
论文地址：https://arxiv.org/abs/2305.10601 \
项目地址：https://github.com/kyegomez/tree-of-thoughts
* 3、多模态KG如何持续学习？浙大等提出首个《持续多模态知识图谱构建》框架 https://mp.weixin.qq.com/s/jwU3Mtdn_2foM65kXRZmaA

# 5.24 周3
* 1、斯坦福transformer课程 https://web.stanford.edu/class/cs25/
* 2、https://github.com/w-okada/voice-changer 实时变声器
* 3、启真医学大模型：利用启真医学知识库构建的中文医学指令数据集，并基于此在LLaMA-7B模型上进行指令精调：An Open Source Chinese Medical Large Language Model' CMKRG GitHub: github.com/CMKRG/QiZhenGPT 
* 4、刚刚！斯坦福发布 AlpacaFarm (羊驼农场)，可将RLHF人工成本降低45倍！(开源) https://mp.weixin.qq.com/s/CIF2F5Vx_RSN1-LwU_ppOQ \
《AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback》 \
Paper：https://tatsu-lab.github.io/alpaca_farm_paper.pdf \
Code：https://github.com/tatsu-lab/alpaca_farm

# 5.25 周4
* 1、chatPaper https://github.com/kaixindelele/ChatPaper
* 2、大模型知识Out该怎么办？浙大团队探索大模型参数更新的方法—模型编辑 https://mp.weixin.qq.com/s/Go_lBw77ktHuHz7FsUjY-w \
![image](https://github.com/shuishenbushui/AICongyin-LLM.github.io/assets/45891944/b8117b05-d35b-4498-95e8-d634146ded0f) \
论文题目：Editing Large Language Models: Problems, Methods, and Opportunities \
论文链接：https://arxiv.org/pdf/2305.13172.pdf
* 3、读脑术！由大脑信号构建高清视频的方法实现啦，Stable Dinfusion还能这么用 https://mp.weixin.qq.com/s/sX0NxL7qQbDSd3H77JzxPw \
参考资料：
 [1]https://mind-video.com/ \
 [2]https://twitter.com/ZijiaoC/status/1660470518569639937 \
 [3]https://arxiv.org/abs/2305.11675 \

# 5.26 周5
* 1、ExpertLLaMA:一个使用ExpertPrompting构建的开源聊天机器人，提供了方法简介、52,000个专家数据集样本、52,000个基线数据集样本、52,000个对应每个具体指令的专家身份描述、基于专家数据集训练的ExpertLLaMA检查点以及与Vicuna、LLaMA-GPT4等现有模型的评估结果github.com/OFA-Sys/ExpertLLaMA
* 2、一个在线的代码生成器，输入自然语言，自动生成代码
网址： www.programming-helper.com/generate-function
* 3、Gorilla: 一个基于LLaMA，为调用API而微调的模型。 \
论文地址：arxiv.org/abs/2305.15334 \
项目地址：gorilla.cs.berkeley.edu \
github:https://github.com/ShishirPatil/gorilla \
![070bb064801d2b5cb9f952a7e3d1549](https://github.com/shuishenbushui/AICongyin-LLM.github.io/assets/45891944/a4743fff-afb3-43f8-8273-aeb244228c76)
* 4、《“According to ...” Prompting Language Models Improves Quoting from Pre-Training Data》 引导LLM根据先前观察到的文本来回答问题 \
![4a1bdf4c8bf922b641ee44c62c6a3a0](https://github.com/shuishenbushui/AICongyin-LLM.github.io/assets/45891944/12359e74-30c7-4a0a-857b-937a8f015be0)

# 5.27 周6
* 1、使用BERT和GPT-2计算句子困惑度PPL ，这个用法可以用来评估数据质量 \
对应代码地址：https://github.com/xu-song/bert-as-language-model
* 2、我用GPT搭建了一个虚拟女友！https://mp.weixin.qq.com/s/8eKNKVZuscejT1qSNIJCig \
1. 作者知乎：https://www.zhihu.com/people/yong-tan-39-67 \
2.我用GPT搭建了一个虚拟女友-哔哩哔哩：https://b23.tv/GYYwMcq
* 3、「大一统」大模型论文爆火，4种模态任意输入输出，华人本科生5篇顶会一作，网友：近期最不可思议的论文 https://mp.weixin.qq.com/s/Mg_qnawkYSWnRHk4LIEIsQ
论文地址：https://arxiv.org/abs/2305.11846 \
项目地址：https://github.com/microsoft/i-Code/tree/main/i-Code-V3
* 4、商汤、清华发布通才智能体完全解锁《我的世界》，像人类一样生存，探索和创造 https://mp.weixin.qq.com/s/GCxvEddxsxSTC3U07KdIag \
"Ghost in the Minecraft"（GITM）\
项目主页：https://github.com/OpenGVLab/GITM
* 5、英伟达AI智能体接入GPT-4，完胜AutoGPT！自主写代码独霸我的世界，无需人类插手 https://mp.weixin.qq.com/s/jaUeCl5pSs-sier89MXq1Q \
论文地址：https://arxiv.org/abs/2305.16291 \
项目地址：https://voyager.minedojo.org/

# 5.28 周7
* 1、InstructScore ，An amazing explanation metric (diagnostic report) for text generation evaluation。\
https://github.com/xu1998hz/SEScore3/
* 2、《A PhD Student's Perspective on Research in NLP in the Era of Very Large Language Models》讨论了未来NLP的可研究方向，汇聚了很多博士生的意见，非常值得读。科研工作者必看！
* 3、终于 ！中文基座模型CPM-Bee开源了 https://mp.weixin.qq.com/s/BO4cDB9KRSODZw3TvZpUAA
* 4、哈工大博士历时半年整理的《Pytorch常用函数函数手册》开放下载！内含200余个函数! https://mp.weixin.qq.com/s/qNAHZzOrVx0kyIeN8AWM9g
