# AICongyin-LLM.github.io
AGI LLM 调研资料汇总

5.7 周7 及之前
>1、ChatGPT 调研报告（仅供内部参考）  哈尔滨工业大学自然语言处理研究所（HIT-NLP）2023年3月6日
>2、Open Chat Video Editor：开源的短视频生成和编辑工具，Python 在线聊天视频编辑器，能实现快速将聊天记录转换为视频格式，支持自定义主题风格和配乐，并能输出多种视频格式。包含了丰富的 API 和示例代码，方便开发者进行二次开发和集成: github.com/SCUTlihaoyu/open-chat-video-editor
3、ChatZoo：对话语言模型横向对比工具。可以将自有模型或者来自 🤗huggingface 的模型轻松部署到网页中。ChatZoo 还可以将一句提示同时发送到多个模型中进行回答生成，方便地对比模型效果： github.com/OpenLMLab/ChatZoo
4、VardaGPT：在GPT-2模型基础上添加了一种称为关联记忆(associative memory)的机制。关联记忆可以使模型在处理文本时更好地捕捉上下文信息。: github.com/ixaxaar/VardaGPT
5、Toolformer：通过prompt令语言模型生成一些能够调用工具的样本，然后利用语言模型对文本的困惑度来过滤这些样本，得到高质量的数据集。最后，利用高质量数据集微调语言模型，赋予其使用工具的能力。
6、ART：人工构建一下常见的工具使用样本，然后利用新任务与已有任务的相似性来辅助语言模型使用工具。
7、InferLLM：非常轻量的 LLM 模型推理框架，可以本地部署 LLM 中的量化模型，推理速度还不错。: github.com/MegEngine/InferLLM
8、用RedPajama基础数据集训练的RedPajama-INCITE模型推出，有3B和7B两个版本，包括指令微调和聊天的版本：https://www.together.xyz/blog/redpajama-models-v1
9、Open LLMs：可供商业使用的开放大型语言模型(LLM)列表github.com/eugeneyan/open-llms
10、Auto-evaluator：文档问答的自动评估: github.com/langchain-ai/auto-evaluator
11、MosaicML NLP团队发布了一个开源的、可商用的语言模型MPT-7B，效果对标LLaMA-7 B。同时发布的还有基于此模型的三个微调的变体，特别是MPT-7B-StoryWriter-65k+ ：一个设计用来阅读和编写超长上下文长度的故事的模型。上下文长度为65 k：https://www.mosaicml.com/blog/mpt-7b
12、Fixing Hallucination with Knowledge Bases  https://www.pinecone.io/learn/langchain-retrieval-augmentation/


5.8 周1
1、Plan，Eliminate,and Track -Language Models are Good Teachers for Embodied Agents. autogpt系列的另外一个工作
2、Donut 🍩 : Document Understanding Transformer
3、多模态语言-图像数据集、LLaVA模型及在线Demo：利用语言模型生成多模态语言-图像指令遵循数据，并用这些数据训练出大型多模态模型LLaVA，用于通用的视觉和语言理解。用语言模型GPT-4生成多模态指令遵循数据，并在HuggingFace Dataset上公开了15.8万条样本；将预训练的CLIP ViT-L/14视觉编码器和大型语言模型LLaMA连接起来，并采用了两阶段的指令微调过程：http://aicoco.net/s/2u
4、5月6日，ChatGPT母公司OpenAI发布了最新开源项目Shap-E，通过文本就能生成3D模型。目前github已经突破2000颗星。（开源地址：https://github.com/openai/shap-e）

5.9 周2
1、ICLR 2023 | DIFFormer: 扩散过程启发的Transformer
本⽂介绍⼀项近期的研究⼯作，试图建⽴能量约束扩散微分⽅程与神经⽹络架构的联系，从而原创性的提出了物理启发下的 Transformer，称作 DIFFormer。作为⼀种通⽤的可以灵活⾼效的学习样本间隐含依赖关系的编码器架构，DIFFormer 在各类任务上都展现了强大潜⼒。
2、SmartGPT https://mp.weixin.qq.com/s/rlu1we8YPJZcu_13iB9obg
能让ChatGPT完成复杂任务，GPT3.5和GPT-4都支持。
它通过将问题拆解，并调用外部资源，提高了GPT的工作能力。
在它的调教下，GPT-4回答的准确率从68%提高到了85%。
3、ChatGPT-Developer-Plugins：不需要 Plus 订阅直接运行 ChatGPT 插件的开发者工具。提供了多种插件，包括翻译、摘要、新闻、笑话等。用户可通过简单的命令行操作使用这些插件，并获取其输出。该工具用 Python 编写，免费提供给所有人使用: github.com/SamurAIGPT/ChatGPT-Developer-Plugins
4、LLaMA Generative Agent：针对LLaMA模型的生成式智能体实现，派生自langchain实现。旨在实现一种生成文本的机制，可用于生成各种文本，包括但不限于电子邮件、文章和评论等。: github.com/UranusSeven/llama_generative_agent
5、Langcorn：用 LangChain + FastApi 部署自动化LLM应用，以便将 LangChain 应用作为 API 提供服务: github.com/msoedov/langcorn
6、ProgGP：包含173首前卫金属风格歌曲的数据集，包括 GuitarPro 格式和Token格式，符合 DadaGP 规范。该数据集可以帮助研究人员和音乐家进行音乐分析和生成相关研究: github.com/otnemrasordep/ProgGP
7、Few-shot In-context Learning for Knowledge Base Question Answering  KBQA

5.10 周3
1、【OpenBuddy：一款强大的开源多语言聊天机器人模型，目标是全球用户，重点是对话AI和流畅的多语言支持，包括英文、中文等多种语言。基于Facebook的LLAMA模型，进行了微调，包括扩展词汇表、增加常用字符和增强的token embeddings。通过这些改进和多轮对话数据集，OpenBuddy提供了一个强大的模型，能回答问题并在各种语言之间进行翻译任务。: github.com/OpenBuddy/OpenBuddy
2、LLM-Leaderboard：由社区联合创建的用于展示大型语言模型(LLM)的集中式排行榜: github.com/LudwigStumpp/llm-leaderboard
3、InfiniteGPT：一个Python脚本，允许您向OpenAI API输入无限大小的文本，从而消除了在使用大量文本输入或将无尽的文本块复制粘贴到chatGPT时重新提示的需求: github.com/emmethalm/infiniteGPT。
4、PaLM的开源复现。已训练了三种不同大小的PaLM模型（150m，410m，1b），并且还在训练一个2b模型: github.com/conceptofmind/PaLM
5、HugNLP：基于Hugging Face Transformer的统一和全面的自然语言处理(NLP)库，旨在提高NLP研究人员的便利性和有效性。该库提供了如BERT、RoBERTa、GPT-2等流行的transformer-based模型，以及一种名为KP-PLM的知识增强预训练范式。HugNLP还实现了一些针对特定任务的模型，包括序列分类、匹配、标注、span提取、多选择和文本生成。此外，该库还支持少样本学习环境，提供了一个原型网络，用于少样本文本分类和命名实体识别(NER)。: github.com/HugAILab/HugNLP
