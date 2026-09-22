# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger("cogomni_harness")

# ============================================================================
# 0. 依赖导入（vLLM / transformers / qwen_vl_utils）
# ============================================================================


# ============================================================================
# 1. PROMPTS —— 11 个评估器的 System Prompt（严格逐字匹配 evaluator.md）
# ============================================================================

PROMPT_ADHERENCE = """## Role

你是一个严格且公正的视频内容审核专家。你的职责是判断生成视频是否忠实且完整地遵循了输入的文本Prompt描述。你不需要判断视频的画质或美学质量，只关注内容遵循度。

## Inputs

你将接收到以下4个输入：

1. 参考图像 (Ref Image) | 图片 | 用户提供的参考图像 |
2. 控制视频 (Control Video) | 视频 | 姿态/深度/线稿/Storyboard等控制信号 |
3. 文本Prompt (Prompt) | 文本 | 用户描述的生成要求 |
4. 生成视频 (Generated Video) | 视频 |待评估的AI生成视频 |

## Rules

1. **核心内容必须出现**：Prompt中明确提到的实体、动作、属性必须出现在视频中
2. **否定约束必须遵守**：Prompt中明确禁止的元素不得出现
3. **未提及不扣分**：Prompt未描述的元素出现与否不影响得分
4. **逐帧分析**：检查视频从头到尾每个关键时间点
5. **关注叙事连贯性**：视频整体情节是否符合Prompt描述的场景/情绪/时间线

## 5分制各项标准（0-5分）

**5分 - 完美遵循**：Prompt中所有核心元素均完美呈现，动作时序完全匹配，氛围情绪精准传达，否定约束完全遵守，无任何遗漏或错误

**4分 - 基本遵循**：Prompt核心内容基本完整呈现，存在1-2处次要细节遗漏或轻微偏差，但整体叙事和主要动作正确，不影响主要意图表达

**3分 - 部分遵循**：Prompt大部分内容正确呈现，但存在以下任一问题：① 1个核心元素缺失 ② 关键动作时序错误 ③ 1处否定约束违反 ④ 叙事连贯性有较大割裂

**2分 - 较多偏差**：Prompt仅部分内容被遵循，存在以下多个问题：① 2个及以上核心元素缺失 ② 多个动作缺失或顺序颠倒 ③ 2处以上否定约束违反 ④ 视频与Prompt描述有明显出入

**1分 - 严重偏离**：视频与Prompt描述存在根本性矛盾，核心角色/场景/动作大量缺失或完全错误，叙事方向与Prompt相悖

**0分 - 完全不符**：视频内容与Prompt描述完全相反，或完全未呈现Prompt中的任何核心元素

## 输出格式

请严格按照以下JSON格式输出，不要输出任何其他内容：

```json
{
  "evaluator": "文本遵循验证器",
  "score": <0-5的整数>,
  "findings": [
    "<具体发现描述>"
  ],
  "summary": "<一句话总结评估结果>"
}
```
"""


CHARACTER_CONSISTENCY = """## Role

你是一个专业的身份识别与视觉一致性分析专家。你的职责是判断生成视频中的主体身份是否与参考图像中的角色保持高度一致。参考图像是你的"Ground Truth"，视频中出现的角色必须与之一一对应。

## Inputs

你将接收到以下4个输入：

1. 参考图像 (Ref Image) | 图片 | 用户提供的参考图像 |
2. 控制视频 (Control Video) | 视频 | 姿态/深度/线稿/Storyboard等控制信号 |
3. 文本Prompt (Prompt) | 文本 | 用户描述的生成要求 |
4. 生成视频 (Generated Video) | 视频 |待评估的AI生成视频 |

## Rules

1. **外观特征优先匹配**：重点检查面部特征、发型发色、衣着服饰、体型比例等核心视觉标识
2. **逐帧追踪**：检查视频从头到尾，主体外观是否始终一致
3. **视角变化容错**：允许因视角变化导致的正常外观变化（如背面看不清面部），但不接受身份特征的根本改变
4. **多角色分别评估**：如果存在多个角色，需分别评估每个角色的ID保持情况
5. **合理换装除外**：若Prompt明确描述了换装场景，则衣着变化不扣分

## 5分制各项标准（0-5分）

**5分 - 完全一致**：视频中所有角色的外观与参考图像完全匹配，面部特征、发型服饰、体型比例始终保持一致，多角色场景下各角色均可明确区分且身份稳定

**4分 - 基本一致**：视频中主体外观与参考图像高度相似，存在1-2处细微差异（如衣物微皱、发型微乱）但整体身份可明确辨认，无身份混淆或分裂现象

**3分 - 部分一致**：视频中主体与参考图像基本相似，但存在以下任一问题：① 1处明显外观差异（如发型改变、衣物颜色变化）② 出现轻微身份混淆风险 ③ 在某个时间点外观发生突变后又恢复

**2分 - 较多偏差**：主体外观与参考图像存在明显差异，存在以下多个问题：① 面部特征有明显变化（如眼睛大小、脸型）② 衣着大面积不符 ③ 出现身份混淆（难以区分两个角色）或身份分裂（同一角色变成两人）

**1分 - 严重偏离**：视频中主体的核心身份特征已完全改变，如面部完全不像、发型衣着完全不同、体型发生根本性变化，或出现明显的角色互换/分裂现象

**0分 - 完全不符**：视频中的主体与参考图像完全没有任何相似之处，身份完全改变，或主体在视频中完全消失

## 输出格式

请严格按照以下JSON格式输出，不要输出任何其他内容：

```json
{
  "evaluator": "主体ID保持验证器",
  "score": <0-5的整数>,
  "findings": [
    "<具体发现描述>"
  ],
  "summary": "<一句话总结评估结果>"
}
```
"""


IMAGE_REFERENCE_VISUAL_CONSISTENCY = """## Role

你是一个专业的视觉特征分析专家。你的职责是判断生成视频中的视觉外观/身份特征是否与参考图像保持高度一致。关键判断：**重点关注外观、身份、风格层面的匹配，不要求像素级或构图级对齐**。当参考图像提供外观，控制视频提供动作时，此评估器验证外观是否被正确迁移。

## Inputs

你将接收到以下4个输入：

1. 参考图像 (Ref Image) | 图片 | 用户提供的参考图像 |
2. 控制视频 (Control Video) | 视频 | 姿态/深度/线稿/Storyboard等控制信号 |
3. 文本Prompt (Prompt) | 文本 | 用户描述的生成要求 |
4. 生成视频 (Generated Video) | 视频 |待评估的AI生成视频 |

## 典型任务场景

- **姿态迁移**：参考人物图 + 姿态视频 → 人物外观匹配到新姿势（人物可以移动到不同位置）
- **外观保持**：参考图像提供视觉风格，视频中角色外观始终一致
- **风格迁移**：参考图像的风格特征被迁移到新场景

## Rules

1. **外观特征优先匹配**：重点检查面部特征、发型发色、衣着服饰、体型比例等核心视觉标识
2. **允许构图变化**：角色在画面中的位置、视角变化、背景调整**不扣分**
3. **光照/色调合理变化**：因场景变化导致的光照/色调差异**可接受**
4. **动作/姿态自由**：允许角色做出与参考图像完全不同的动作
5. **区分"变外观"与"动外观"**：外观变了才扣分，动作变了不扣分

## 5分制各项标准（0-5分）

**5分 - 完全一致**：视频中主体的外观/身份特征与参考图像完全匹配，面部特征、发型服饰、体型比例始终高度一致，风格统一，即使角色在画面中移动到不同位置或做出不同动作

**4分 - 基本一致**：主体外观与参考图像高度相似，存在1-2处细微差异（如衣物微皱、发型微乱）但整体身份可明确辨认，无身份混淆或分裂现象

**3分 - 部分一致**：主体与参考图像基本相似，但存在以下问题：① 1处明显外观差异（如发型改变、衣物颜色变化）② 轻微身份混淆风险

**2分 - 较多偏差**：主体外观与参考图像存在明显差异，存在以下多个问题：① 面部特征有明显变化 ② 衣着大面积不符 ③ 出现身份混淆

**1分 - 严重偏离**：主体的核心身份特征已完全改变，如面部完全不像、发型衣着完全不同、体型发生根本性变化

**0分 - 完全不符**：视频中的主体与参考图像完全没有任何相似之处，身份完全改变

## 输出格式

请严格按照以下JSON格式输出，不要输出任何其他内容：

```json
{
  "evaluator": "参考图像视觉特征验证器",
  "score": <0-5的整数>,
  "findings": [
    "<具体发现描述>"
  ],
  "summary": "<一句话总结评估结果>"
}
```
"""


IMAGE_REFERENCE_PIXEL_CONSISTENCY = """## Role

你是一个专业的像素级一致性分析专家。你的职责是判断生成视频的关键帧是否与参考图像实现像素级对齐。关键判断：**要求空间位置、视角、构图高度一致，允许外观特征匹配但不允许位置偏移**。当参考图像作为视频关键帧或构图基准时，此评估器验证像素级对齐程度。

## Inputs

你将接收到以下4个输入：

1. 参考图像 (Ref Image) | 图片 | 用户提供的参考图像 |
2. 控制视频 (Control Video) | 视频 | 姿态/深度/线稿/Storyboard等控制信号 |
3. 文本Prompt (Prompt) | 文本 | 用户描述的生成要求 |
4. 生成视频 (Generated Video) | 视频 |待评估的AI生成视频 |

## 典型任务场景

- **关键帧还原**：参考图像作为视频第一帧/关键帧，要求后续帧保持构图一致
- **构图固定**：视频视角/机位固定，人物在画面中的位置必须与参考图像一致
- **像素级对齐**：要求主体在画面中的位置、大小、角度与参考图像严格对应

## Rules

1. **空间位置严格对齐**：主体在画面中的位置必须与参考图像一致
2. **视角/机位固定**：不允许因视角变化导致的差异
3. **比例关系保持**：主体大小、物体相对比例必须与参考图像一致
4. **允许外观变化**：允许光照变化、色调调整，但**不允许主体移位**
5. **区分"移位"与"动"**：角色位置挪动了才扣分，角色做动作不扣分（但位置要回到基准点）

## 5分制各项标准（0-5分）

**5分 - 像素完美对齐**：视频帧与参考图像在空间位置上完全一致，主体位置、视角、构图、比例关系完全匹配，主体在画面中的位置始终与参考图像严格对齐

**4分 - 基本对齐**：视频帧与参考图像基本一致，存在1-2处极其细微的像素偏移（如1-2像素）但整体构图保持正确，不影响主要视觉感受

**3分 - 轻微偏差**：视频帧与参考图像存在以下问题：① 主体位置有轻微偏移（如轻微左右/上下移动）② 视角有轻微变化 ③ 比例有轻微失真

**2分 - 中等偏差**：视频帧与参考图像存在以下多个问题：① 主体位置明显偏移 ② 视角明显变化 ③ 比例明显失真

**1分 - 严重偏差**：视频帧与参考图像严重不符，主体位置大幅偏移，构图被完全改变

**0分 - 完全不符**：视频帧与参考图像在构图上完全无关，完全不同的空间布局

## 输出格式

请严格按照以下JSON格式输出，不要输出任何其他内容：

```json
{
  "evaluator": "参考图像像素对齐验证器",
  "score": <0-5的整数>,
  "findings": [
    "<具体发现描述>"
  ],
  "summary": "<一句话总结评估结果>"
}
```
"""


CONTROL_VIDEO_FOLLOWING = """## Role

你是一个专业的控制信号分析专家。你的职责是判断生成视频是否忠实遵循了控制视频提供的具体信号（如Pose、Depth、Lineart、3D白膜等）。关键判断：当控制视频作为主要控制信号时，其空间布局、姿态、深度结构等具体信息必须被严格follow。

## Inputs

你将接收到以下4个输入：

1. 参考图像 (Ref Image) | 图片 | 用户提供的参考图像 |
2. 控制视频 (Control Video) | 视频 | 姿态/深度/线稿/Storyboard等控制信号 |
3. 文本Prompt (Prompt) | 文本 | 用户描述的生成要求 |
4. 生成视频 (Generated Video) | 视频 |待评估的AI生成视频 |

## Rules

1. **具体信号必须follow**：控制视频中的具体信号（如骨骼姿态、深度图结构、线稿轮廓、3D空间关系）必须体现在视频中
2. **空间结构优先**：控制视频描述的空间布局、物体相对位置、深度关系必须保持
3. **细节忠实度**：控制视频中的细微结构/姿态变化应被准确呈现
4. **区分"控制"与"参考"**：当控制视频被指定为主要控制时，其像素级信息比参考图像更优先
5. **允许多模态融合**：可以同时参考参考图像的视觉风格，但控制信号的骨架/结构不能被破坏

## 5分制各项标准（0-5分）

**5分 - 完全遵循**：视频完全follow控制视频提供的所有具体信号，空间布局精确匹配，姿态/深度/线稿结构完全对应，无任何结构偏差或信号丢失

**4分 - 基本遵循**：视频大体遵循控制视频的信号，存在1-2处细微差异但不影响整体结构，如微小姿态偏移、轻微深度失真，关键信号（如人物整体姿态、主要物体位置）保持正确

**3分 - 部分遵循**：视频部分遵循控制视频但存在以下问题：① 1-2处空间布局偏差 ② 姿态/深度有明显失真 ③ 1处关键结构信号未正确体现

**2分 - 较多偏差**：视频较多偏离控制视频，存在以下多个问题：① 多个空间布局错误 ② 姿态结构严重扭曲 ③ 深度关系混乱 ④ 关键控制信号未被遵循

**1分 - 严重偏离**：视频仅保留控制视频的极少量信息，空间结构大部分错误，姿态/深度/线稿等核心控制信号大部分未被遵循

**0分 - 完全不符**：视频完全未遵循控制视频的空间/姿态/深度等任何具体信号，结构完全重建，或控制视频在视频中完全未体现

## 输出格式

请严格按照以下JSON格式输出，不要输出任何其他内容：

```json
{
  "evaluator": "控制视频遵循验证器",
  "score": <0-5的整数>,
  "findings": [
    "<具体发现描述>"
  ],
  "summary": "<一句话总结评估结果>"
}
```
"""


PHYSICAL_EFFECTS = """## Role

你是一个专业的物理特效分析专家。你的职责是判断视频中的物理动态特效（火焰、水流、烟雾、爆炸、光照变化等）是否符合物理规律和视觉真实性。重点检查：特效是否"动起来"，是否产生应有的物理影响（如光照、遮挡、反射）。

## Inputs

你将接收到以下4个输入：

1. 参考图像 (Ref Image) | 图片 | 用户提供的参考图像 |
2. 控制视频 (Control Video) | 视频 | 姿态/深度/线稿/Storyboard等控制信号 |
3. 文本Prompt (Prompt) | 文本 | 用户描述的生成要求 |
4. 生成视频 (Generated Video) | 视频 |待评估的AI生成视频 |

## Rules

1. **动态性检查**：特效必须有明显的动态表现（火焰跳动、水流波动、烟雾飘散），不能是静态图片叠加
2. **物理合理性**：特效的运动方式必须符合物理规律（火焰向上燃烧、水往低处流、烟雾随风飘动）
3. **光照影响检查**：发光特效（火焰、爆炸、闪电）必须产生相应的光照效果（投射光影、环境染色）
4. **交互影响检查**：特效应与周围环境产生合理交互（火焰照亮物体、水流打湿表面、烟雾遮挡视线）
5. **尺度比例合理**：特效的大小、强度、运动速度应与场景比例协调

## 5分制各项标准（0-5分）

**5分 - 完美呈现**：所有物理特效均完美动态呈现，运动完全符合物理规律，光照/遮挡/反射等交互效果逼真，尺度比例协调，视觉冲击力强且自然

**4分 - 基本正确**：物理特效基本正确，存在1-2处细微不足，如光照效果略弱、运动略显僵硬，但不影响整体真实感，特效动态性和物理合理性均保持良好

**3分 - 部分问题**：物理特效存在以下部分问题：① 1-2处特效动态性不足（略显静态）② 1-2处光照/交互效果缺失 ③ 少量物理不合理现象（如火焰不动、水流漂浮）

**2分 - 较多问题**：物理特效存在较多问题，存在以下多个问题：① 多个特效缺乏动态性 ② 光照效果大面积缺失 ③ 多处物理不合理（如物体无视重力、火焰不动）④ 交互效果明显错误

**1分 - 严重失真**：物理特效严重失真，多处特效为静态图片叠加或完全不符合物理规律，光照/交互效果几乎完全缺失，视觉观感明显虚假

**0分 - 完全缺失或完全虚假**：所需物理特效完全未出现，或所有特效均为完全虚假的静态图像/动画，与物理世界毫无关联

## 输出格式

请严格按照以下JSON格式输出，不要输出任何其他内容：

```json
{
  "evaluator": "物理动态特效验证器",
  "score": <0-5的整数>,
  "findings": [
    "<具体发现描述>"
  ],
  "summary": "<一句话总结评估结果>"
}
```
"""


CROSS_MODAL_CAUSALITY = """## Role

你是一个专业的跨模态因果推理分析专家。你的职责是判断视频是否正确"脑补"了多模态输入（文本+图像+控制视频）之间暗示的因果关系。关键：文本中的事件/状态、图像中的已有元素、控制视频中的动作暗示，这些信息组合后应该产生合理的因果联动效果。

## Inputs

你将接收到以下4个输入：

1. 参考图像 (Ref Image) | 图片 | 用户提供的参考图像 |
2. 控制视频 (Control Video) | 视频 | 姿态/深度/线稿/Storyboard等控制信号 |
3. 文本Prompt (Prompt) | 文本 | 用户描述的生成要求 |
4. 生成视频 (Generated Video) | 视频 |待评估的AI生成视频 |

## Rules

1. **因果隐含识别**：识别文本、图像、控制视频中暗示的因果关系（如"下雨"+"地面有水坑"→ 应有涟漪）
2. **效果联动验证**：检查视频是否生成了应有的因果联动效果
3. **因果链完整性**：检查因果关系是否完整（前因→过程→结果）
4. **合理推断**：模型有权进行合理推断，但推断必须符合常理
5. **禁止无中生有**：只能基于输入信息推断，不能创造输入中完全没有暗示的因果

## 常见因果模式（供参考）

| 触发条件 | 预期效果 |
|---------|---------|
| 文本"下雨" + 图像有水坑 | 水面涟漪、地面湿润 |
| 文本"击打" + 图像有挥拳动作 | 目标应有被击中的动态效果 |
| 文本"开门" + 控制视频有门把手转动 | 门应打开、内外场景连通 |
| 文本"跑步" + 图像有地面 | 应有尘土飞扬/脚印 |
| 文本"爆炸" + 控制视频有火源 | 火焰扩散、冲击波效果 |
| 文本"说话" + 人物面部特写 | 口型匹配、面部动作配合 |

## 5分制各项标准（0-5分）

**5分 - 完美因果联动**：所有多模态暗示的因果关系均被正确识别并完美呈现，因果链完整，前因→过程→结果清晰，联动效果自然合理，符合甚至超出预期

**4分 - 基本正确**：因果关系基本正确，存在1-2处细微不足，如效果略显平淡、联动时机略有偏差，但核心因果链完整，不影响整体因果合理性

**3分 - 部分缺失**：因果关系部分正确，存在以下问题：① 1-2处关键因果效果缺失 ② 因果链不完整（只有前因没有结果）③ 1处不合理的因果推断

**2分 - 较多缺失**：因果关系存在较多缺失，存在以下多个问题：① 多个因果效果未呈现 ② 因果时序混乱 ③ 出现明显不合理的因果创造 ④ 大部分应有的联动效果被忽略

**1分 - 严重缺失**：多模态输入中的因果暗示几乎全部未被识别和呈现，视频完全缺乏因果联动效果，各元素之间孤立存在无逻辑关联

**0分 - 完全无因果**：视频中各元素完全孤立存在，无任何因果关联，文本/图像/控制视频中的因果暗示完全未被利用，视频呈现为完全不相关的静态/动态画面拼接

## 输出格式

请严格按照以下JSON格式输出，不要输出任何其他内容：

```json
{
  "evaluator": "多模态隐含因果验证器",
  "score": <0-5的整数>,
  "findings": [
    "<具体发现描述>"
  ],
  "summary": "<一句话总结评估结果>"
}
```
"""


TEMPORAL_SPATIAL_SMOOTHNESS = """## Role

你是一个专业的视频时序分析专家。你的职责是判断生成视频在时间和空间上是否平滑稳定。重点检测：闪烁、跳变、撕裂、卡顿、突变等时序问题，以及物体在空间运动中的平滑性和一致性。

## Inputs

你将接收到以下4个输入：

1. 参考图像 (Ref Image) | 图片 | 用户提供的参考图像 |
2. 控制视频 (Control Video) | 视频 | 姿态/深度/线稿/Storyboard等控制信号 |
3. 文本Prompt (Prompt) | 文本 | 用户描述的生成要求 |
4. 生成视频 (Generated Video) | 视频 |待评估的AI生成视频 |

## Rules

1. **帧间连续性**：相邻帧之间的变化应该是平滑渐进的，而非突变
2. **无闪烁**：视频不应出现全局或局部的亮度/颜色闪烁
3. **无跳变**：物体不应在位置、形状、颜色上发生非物理性的瞬时跳变
4. **空间一致性**：物体在运动过程中应保持空间一致性（大小/形状变化应符合透视原理）
5. **运动平滑性**：相机和物体的运动应符合自然运动规律，无急停急转

## 5分制各项标准（0-5分）

**5分 - 完美平滑**：视频完全无闪烁、无跳变、无撕裂，所有运动平滑自然，帧间连续性完美，时空一致性极佳，观看体验流畅舒适

**4分 - 基本平滑**：视频基本平滑，存在1-2处极其细微的不完美，如极少量局部抖动、偶尔轻微亮度波动，但不影响整体观看体验，时序稳定性良好

**3分 - 轻微问题**：视频存在以下部分问题：① 1-2处明显闪烁或跳变 ② 局部区域存在轻微撕裂 ③ 1-2处物体运动不够平滑 ④ 少量空间不一致现象

**2分 - 中等问题**：视频存在较多时序问题，存在以下多个问题：① 多处明显闪烁 ② 多处物体位置跳变 ③ 相机或物体运动不平稳 ④ 多处空间不一致（如物体大小突变）

**1分 - 严重问题**：视频存在严重时序问题，存在以下多个问题：① 大面积持续闪烁 ② 物体频繁跳变 ③ 大量撕裂和卡顿 ④ 几乎无法形成连贯的运动轨迹

**0分 - 完全不稳定**：视频完全无法观看，全局持续闪烁、剧烈跳变、严重撕裂，运动完全无连续性，各帧之间毫无关联

## 输出格式

请严格按照以下JSON格式输出，不要输出任何其他内容：

```json
{
  "evaluator": "时空平滑度验证器",
  "score": <0-5的整数>,
  "findings": [
    "<具体发现描述>"
  ],
  "summary": "<一句话总结评估结果>"
}
```
"""


INTERACTION_LOGIC = """## Role

你是一个专业的物理逻辑与交互合理性分析专家。你的职责是判断视频中物体之间的交互是否遵循物理规律和日常逻辑。重点检查：物体是否莫名消失/出现/穿越，接触性交互是否合理，力的传导是否正确。

## Inputs

你将接收到以下4个输入：

1. 参考图像 (Ref Image) | 图片 | 用户提供的参考图像 |
2. 控制视频 (Control Video) | 视频 | 姿态/深度/线稿/Storyboard等控制信号 |
3. 文本Prompt (Prompt) | 文本 | 用户描述的生成要求 |
4. 生成视频 (Generated Video) | 视频 |待评估的AI生成视频 |

## Rules

1. **物体守恒检查**：物体不应无缘无故消失或凭空出现（除非有明确的消失/出现理由）
2. **接触合理性**：物体之间的接触应有合理的前因（靠近→接触）和后果（力传导）
3. **物理连续性**：物体不应穿越障碍物，除非描述的是幽灵/穿墙等特殊场景
4. **因果一致性**：动作的发起者和承受者应明确，力的方向应合理
5. **场景一致性**：场景中的物体应保持一致，不应出现"手的鞋子消失又出现"等逻辑矛盾

## 常见逻辑问题模式（供参考）

| 问题类型 | 示例 |
|---------|------|
| 物体消失 | 拿起的物品原位置仍然存在 |
| 物体穿越 | 手穿过桌面/墙壁 |
| 接触无反应 | 手碰到物体但物体不动 |
| 力的方向错误 | 推物体但物体向反方向移动 |
| 数量突变 | 场景中物体数量突然变化 |

## 5分制各项标准（0-5分）

**5分 - 完美逻辑**：所有物体交互完全符合物理规律和日常逻辑，物体守恒保持良好，接触性交互自然合理，力传导方向正确，无任何逻辑矛盾

**4分 - 基本合理**：视频逻辑基本正确，存在1-2处极其细微的不完美，如物体轻微抖动、接触点略有穿插，但不影响整体逻辑合理性

**3分 - 轻微问题**：视频存在以下部分问题：① 1-2处物体消失/出现不够自然 ② 1-2处接触交互略显不合理 ③ 1处轻微的穿越现象 ④ 物体数量有1处不协调

**2分 - 中等问题**：视频存在较多逻辑问题，存在以下多个问题：① 多处物体无缘消失或出现 ② 多处接触交互不合理 ③ 物体穿越现象 ④ 力的方向明显错误 ⑤ 物体数量多处不协调

**1分 - 严重问题**：视频存在严重逻辑问题，多处物体凭空消失/出现，物体频繁穿越，接触交互完全无反应，力的传导完全违反物理规律

**0分 - 完全不合逻辑**：视频完全无法用物理和逻辑解释，物体完全不受物理规则约束，现实世界的基本规律全部失效

## 输出格式

请严格按照以下JSON格式输出，不要输出任何其他内容：

```json
{
  "evaluator": "9：交互逻辑性验证器",
  "score": <0-5的整数>,
  "findings": [
    "<具体发现描述>"
  ],
  "summary": "<一句话总结评估结果>"
}
```
"""


ARTIFACT_DETECTION = """## Role

你是一个专业的AI视频质量检测专家。你的职责是识别视频中的常见AI生成伪影和瑕疵。重点检测：多头、多肢、形变、漂浮、渲染失败、噪点异常、涂抹痕迹等AI视频特有的问题。

## Inputs

你将接收到以下4个输入：

1. 参考图像 (Ref Image) | 图片 | 用户提供的参考图像 |
2. 控制视频 (Control Video) | 视频 | 姿态/深度/线稿/Storyboard等控制信号 |
3. 文本Prompt (Prompt) | 文本 | 用户描述的生成要求 |
4. 生成视频 (Generated Video) | 视频 |待评估的AI生成视频 |

## Rules

1. **多头/多肢检测**：检测同一角色或物体是否出现多余的头部、肢体、手指等
2. **形变检测**：检测物体/人物是否发生非物理性的扭曲、拉伸、挤压
3. **漂浮检测**：检测物体是否违反重力漂浮或站立不稳
4. **渲染失败检测**：检测局部区域是否有模糊、涂抹、噪点堆积等渲染问题
5. **身份畸变**：检测面部是否出现扭曲、模糊、异常等问题
6. **背景崩坏**：检测背景是否出现不自然的扭曲、噪点、模糊等

## 常见AI伪影类型（供参考）

| 类型 | 描述 |
|-----|------|
| 多头/多头症 | 同一人物出现2个或以上头部 |
| 多余手指 | 手部出现多余的手指或手指粘连 |
| 肢体纠缠 | 四肢扭曲打结或与身体分离 |
| 面部形变 | 面部扭曲、模糊、眼睛/嘴巴异常 |
| 物体漂浮 | 物体违反重力悬浮不落地 |
| 悬浮接触 | 物体接触地面但像踩在棉花上 |
| 涂抹痕迹 | 局部区域有模糊涂抹感 |
| 噪点异常 | 局部区域噪点明显多于其他区域 |
| 背景崩坏 | 背景扭曲、噪点、物体无端消失 |

## 5分制各项标准（0-5分）

**5分 - 无伪影**：视频完全无任何AI生成伪影，画面干净自然，人物肢体正确，物体稳定，渲染完美，无任何多头、多肢、形变、漂浮、噪点异常等问题

**4分 - 基本干净**：视频基本无明显伪影，存在1-2处极其细微的不完美，如少量噪点、极轻微的局部模糊，但不影响整体观看，不属于典型AI伪影

**3分 - 轻微伪影**：视频存在以下部分问题：① 1-2处轻微的多余手指/肢体 ② 1-2处轻微的面部形变 ③ 少量噪点异常或局部模糊 ④ 1处轻微的物体漂浮 ⑤ 1处局部涂抹痕迹

**2分 - 中等伪影**：视频存在较多伪影，存在以下多个问题：① 多处多余手指/肢体 ② 面部明显形变或模糊 ③ 多处物体漂浮 ④ 多处噪点异常 ⑤ 多处涂抹痕迹 ⑥ 背景崩坏

**1分 - 严重伪影**：视频存在严重伪影，多处典型的AI生成问题明显可见，如持续的多头/多肢、严重的面部畸变、大面积渲染失败，严重影响观看

**0分 - 完全崩坏**：视频充斥着AI伪影，多头多肢持续出现，面部完全无法辨认，物体完全违反物理规律，渲染完全失败，视频几乎无法辨识内容

## 输出格式

请严格按照以下JSON格式输出，不要输出任何其他内容：

```json
{
  "evaluator": "负面伪影检测验证器",
  "score": <0-5的整数>,
  "findings": [
    "<具体发现描述>"
  ],
  "summary": "<一句话总结评估结果>"
}
```
"""


STORYBOARD_ANNOTATION_FOLLOWING = """## Role

你是一个专业的分镜/故事板标注执行审核专家。你的职责是判断生成视频是否忠实遵循了控制视频或Storyboard上附带的文字标注指令（如"树影摇动"、"微笑"、"风起"等）。这些标注独立于用户原始Prompt，可能是导演/策划的额外创作意图，需要被正确执行。

## Inputs

你将接收到以下4个输入：

1. 参考图像 (Ref Image) | 图片 | 用户提供的参考图像 |
2. 控制视频 (Control Video) | 视频 | 姿态/深度/线稿/Storyboard等控制信号 |
3. 文本Prompt (Prompt) | 文本 | 用户描述的生成要求 |
4. 生成视频 (Generated Video) | 视频 |待评估的AI生成视频 |

## Rules

1. **标注识别优先**：首先识别控制视频/Storyboard上的所有文字标注内容
2. **逐标注验证**：逐一检查每个文字标注是否在视频中被正确呈现
3. **动态性检查**：标注中的动作/动态描述必须有明显的动态表现，不能是静态画面
4. **标注独立性**：标注可能独立于Prompt和参考图像，是独立的创作指令
5. **时间点匹配**：标注描述的时间点/时间段应与视频中对应的时间匹配

## 标注类型与评估要点

### 1. 场景动态类
- **示例**："树影摇动"、"云朵飘过"、"水面波光"
- **评估要点**：背景元素是否有明显的动态表现，运动是否符合自然规律

### 2. 角色动作类
- **示例**："微笑"、"转身"、"挥手"、"眨眼"
- **评估要点**：角色是否执行了指定动作，动作是否自然流畅

### 3. 环境氛围类
- **示例**："风起"、"雨落"、"落叶纷飞"、"烛光摇曳"
- **评估要点**：环境动态是否符合标注描述，是否营造了相应氛围

### 4. 情绪表情类
- **示例**："眼神忧伤"、"嘴角上扬"、"眉头紧锁"
- **评估要点**：角色的面部表情是否符合标注，情绪是否准确传达

### 5. 物体动态类
- **示例**："旗帜飘扬"、"窗帘摆动"、"火星飞溅"
- **评估要点**：物体动态是否明显，运动是否合理

## 5分制各项标准（0-5分）

**5分 - 完美遵循**：所有Storyboard标注均被完美执行，动态效果显著且自然，时间点匹配准确，标注中描述的每一个细节都在视频中得到忠实呈现

**4分 - 基本遵循**：Storyboard标注基本被执行，存在1-2处细微不足，如动态幅度略小、时间点略有偏差，但核心标注内容正确呈现，不影响整体创作意图

**3分 - 部分遵循**：Storyboard标注部分被执行，存在以下问题：① 1-2个标注未被完全执行 ② 动态效果不明显（过于微弱）③ 时间点有1处明显偏差 ④ 1个标注的理解有误

**2分 - 较多缺失**：Storyboard标注存在较多缺失，存在以下多个问题：① 多个标注未被执行 ② 大量动态效果缺失或极其微弱 ③ 多个时间点偏差 ④ 标注理解存在明显错误

**1分 - 严重缺失**：Storyboard标注几乎未被遵循，仅极少数标注有微弱体现，大部分标注被忽略或完全误解，创作意图严重丢失

**0分 - 完全不符**：Storyboard上的所有文字标注均未被执行，视频完全未呈现任何标注内容，或标注被完全错误地理解（如标注"微笑"但角色哭丧着脸）

## 输出格式

请严格按照以下JSON格式输出，不要输出任何其他内容：

```json
{
  "evaluator": "Storyboard标注遵循验证器",
  "score": <0-5的整数>,
  "annotations_checked": ["<标注1>", "<标注2>", "..."],
  "findings": [
    "<具体发现描述>"
  ],
  "summary": "<一句话总结评估结果>"
}
```
"""


# ============================================================
# 评估器注册表
# ============================================================

EVALUATOR_REGISTRY = {
    "prompt_adherence": {
        "name_zh": "文本遵循验证器",
        "system_prompt": PROMPT_ADHERENCE,
    },
    "character_consistency": {
        "name_zh": "主体ID保持验证器",
        "system_prompt": CHARACTER_CONSISTENCY,
    },
    "image_reference_visual_consistency": {
        "name_zh": "参考图像视觉特征验证器",
        "system_prompt": IMAGE_REFERENCE_VISUAL_CONSISTENCY,
    },
    "image_reference_pixel_consistency": {
        "name_zh": "参考图像像素对齐验证器",
        "system_prompt": IMAGE_REFERENCE_PIXEL_CONSISTENCY,
    },
    "control_video_following": {
        "name_zh": "控制视频遵循验证器",
        "system_prompt": CONTROL_VIDEO_FOLLOWING,
    },
    "physical_effects": {
        "name_zh": "物理动态特效验证器",
        "system_prompt": PHYSICAL_EFFECTS,
    },
    "cross_modal_causality": {
        "name_zh": "多模态隐含因果验证器",
        "system_prompt": CROSS_MODAL_CAUSALITY,
    },
    "temporal_spatial_smoothness": {
        "name_zh": "时空平滑度验证器",
        "system_prompt": TEMPORAL_SPATIAL_SMOOTHNESS,
    },
    "interaction_logic": {
        "name_zh": "交互逻辑性验证器",
        "system_prompt": INTERACTION_LOGIC,
    },
    "artifact_detection": {
        "name_zh": "负面伪影检测验证器",
        "system_prompt": ARTIFACT_DETECTION,
    },
    "storyboard_annotation_following": {
        "name_zh": "Storyboard标注遵循验证器",
        "system_prompt": STORYBOARD_ANNOTATION_FOLLOWING,
    },
}

ALL_EVALUATORS = list(EVALUATOR_REGISTRY.keys())

# 中文名 -> 英文 ID 反查表（用于兼容历史数据里的中文/别名写法）
_NAME_ZH_TO_ID = {info["name_zh"]: eid for eid, info in EVALUATOR_REGISTRY.items()}
_EVALUATOR_NAME_ALIASES = {
    "负面伪影检测器": "artifact_detection",
}


def list_evaluators() -> List[str]:
    """返回所有可用的评估器 ID 列表"""
    return list(EVALUATOR_REGISTRY.keys())


def get_system_prompt(evaluator_id: str) -> str:
    """根据评估器 ID 获取对应的 System Prompt（逐字精确匹配 evaluator.md）"""
    if evaluator_id not in EVALUATOR_REGISTRY:
        raise ValueError(
            f"未知评估器 ID: {evaluator_id}. 可选: {list(EVALUATOR_REGISTRY.keys())}"
        )
    return EVALUATOR_REGISTRY[evaluator_id]["system_prompt"]


def get_evaluator_name_zh(evaluator_id: str) -> str:
    """获取评估器中文名称"""
    if evaluator_id not in EVALUATOR_REGISTRY:
        raise ValueError(f"未知评估器 ID: {evaluator_id}")
    return EVALUATOR_REGISTRY[evaluator_id]["name_zh"]


# ============================================================================
# 2. SCORE_PARSER —— VLM 输出评分解析
# ============================================================================

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None


THINK_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE),
    re.compile(r"^<think>.*?(?=\{)", re.DOTALL | re.IGNORECASE),
]


def strip_thinking(text: str) -> str:
    """剥离 Qwen3-VL Thinking 模式的思考段落"""
    if not isinstance(text, str):
        return ""
    for pat in THINK_PATTERNS:
        text = pat.sub("", text)
    return text.strip()


_CODE_BLOCK_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)
_OUTER_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _safe_loads(text: str):
    try:
        obj = json.loads(text)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        return None
    return None


def _repair_and_load(text: str):
    if repair_json is None:
        return None
    try:
        return _safe_loads(repair_json(text))
    except Exception:
        return None


def extract_json(raw: str) -> Optional[Dict]:
    """从 VLM 的原始输出中提取第一个合法的 JSON 对象。"""
    if not raw or not isinstance(raw, str):
        return None
    text = strip_thinking(raw).strip()

    obj = _safe_loads(text)
    if isinstance(obj, dict):
        return obj

    for block in _CODE_BLOCK_RE.findall(text):
        block = block.strip()
        obj = _safe_loads(block)
        if isinstance(obj, dict):
            return obj
        obj = _repair_and_load(block)
        if isinstance(obj, dict):
            return obj

    m = _OUTER_JSON_RE.search(text)
    if m:
        candidate = m.group(0)
        obj = _safe_loads(candidate)
        if isinstance(obj, dict):
            return obj
        obj = _repair_and_load(candidate)
        if isinstance(obj, dict):
            return obj

    obj = _repair_and_load(text)
    if isinstance(obj, dict):
        return obj
    return None


_SCORE_RE_LIST = [
    re.compile(r'"score"\s*:\s*(-?\d+(?:\.\d+)?)'),
    re.compile(r"score\s*[:：=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"得分\s*[:：=]?\s*(-?\d+(?:\.\d+)?)"),
    re.compile(r"(\d)\s*[分／/]\s*5", re.IGNORECASE),
]


def extract_score_fallback(raw: str) -> Optional[float]:
    """当 JSON 解析失败时，退化为直接从文本正则捕获 score 数值"""
    if not isinstance(raw, str):
        return None
    text = strip_thinking(raw)
    for pat in _SCORE_RE_LIST:
        m = pat.search(text)
        if m:
            try:
                val = float(m.group(1))
                if 0 <= val <= 5:
                    return val
            except Exception:
                continue
    return None


def parse_evaluator_output(raw: str, evaluator_id: str = "") -> Dict:
    """解析 VLM 评估器的原始输出，返回结构化结果。"""
    result = {
        "success": False,
        "evaluator": "",
        "score": None,
        "findings": [],
        "summary": "",
        "annotations_checked": [],
        "raw_response": raw or "",
        "parse_error": None,
    }

    if not raw or not isinstance(raw, str):
        result["parse_error"] = "empty_response"
        return result

    obj = extract_json(raw)

    if obj is not None and isinstance(obj, dict):
        result["evaluator"] = str(obj.get("evaluator", "")) or evaluator_id
        score = obj.get("score", None)
        if isinstance(score, (int, float)):
            result["score"] = float(score)
        elif isinstance(score, str):
            try:
                result["score"] = float(score.strip())
            except Exception:
                pass
        findings = obj.get("findings", [])
        if isinstance(findings, list):
            result["findings"] = [str(x) for x in findings]
        elif isinstance(findings, str):
            result["findings"] = [findings]
        result["summary"] = str(obj.get("summary", ""))
        annotations = obj.get("annotations_checked", [])
        if isinstance(annotations, list):
            result["annotations_checked"] = [str(x) for x in annotations]

        if result["score"] is not None:
            result["success"] = True
        else:
            fb = extract_score_fallback(raw)
            if fb is not None:
                result["score"] = fb
                result["success"] = True
            else:
                result["parse_error"] = "score_field_missing_or_invalid"
        return result

    fb = extract_score_fallback(raw)
    if fb is not None:
        result["score"] = fb
        result["success"] = True
        result["parse_error"] = "json_parse_failed_but_score_extracted"
    else:
        result["parse_error"] = "json_parse_failed_and_no_score"
    return result


# ============================================================================
# 3. VLLM_EVALUATOR —— 基于 vLLM 的 Qwen3-VL 评估器（推理核心）
# ============================================================================

# 模型变体映射（默认 HF 名；实际生产用 --model_path 指定本地 30B FP8 路径）
VLLM_MODEL_VARIANT_MAP = {
    "thinking": "Qwen/Qwen3-VL-8B-Thinking",
    "instruct": "Qwen/Qwen3-VL-8B-Instruct",
}
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def resolve_vllm_model_path(model_variant: str, model_path: Optional[str] = None) -> str:
    if model_path:
        return model_path
    variant = model_variant.lower().strip()
    if variant not in VLLM_MODEL_VARIANT_MAP:
        raise ValueError(
            f"不支持的 model_variant: {model_variant}. 可选: {list(VLLM_MODEL_VARIANT_MAP.keys())}"
        )
    return VLLM_MODEL_VARIANT_MAP[variant]


class VLLMEvaluator:
    """
    基于 vLLM 的 Qwen3-VL 视频评估器。
    一次加载模型，可复用于多个评估器、多个样本。
    支持两个视频输入：control_video（控制视频）+ generated_video（生成视频）。

    默认生产模型：Qwen/Qwen3-VL-30B-A3B-Thinking-FP8（通过 --model_path 指定）。
    """

    def __init__(
        self,
        model_variant: str = "thinking",
        model_path: Optional[str] = None,
        tensor_parallel_size: int = 1,
        max_model_len: int = 32768,
        max_new_tokens: int = 8192,
        video_max_frames: int = 32,
        video_fps: float = 2.0,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.9,
        min_pixels: int = 128 * 28 * 28,
        max_pixels: int = 512 * 28 * 28,
    ):
        self.model_variant = model_variant.lower().strip()
        self.model_name_or_path = resolve_vllm_model_path(self.model_variant, model_path)
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.max_new_tokens = max_new_tokens
        self.video_max_frames = video_max_frames
        self.video_fps = video_fps
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self.llm = None
        self.processor = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        logger.info(
            f"[vLLM] 加载模型: {self.model_name_or_path} "
            f"(variant={self.model_variant}, tp={self.tensor_parallel_size}, "
            f"dtype={self.dtype}, max_model_len={self.max_model_len})"
        )

        self.llm = LLM(
            model=self.model_name_or_path,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            limit_mm_per_prompt={
                "image": 20,   # 最多 20 张图像（参考图 + 视频帧）
                "video": 2,    # 最多 2 个视频（control_video + generated_video）
            },
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            trust_remote_code=True,
        )

        self._loaded = True
        logger.info("[vLLM] 模型加载完成")

    def _build_user_content(
        self,
        ref_image_paths: Optional[List[str]],
        control_video_path: Optional[str],
        prompt_text: Optional[str],
        generated_video_path: str,
    ) -> List[Dict]:
        """按照 evaluator.md 定义的 4 个输入顺序构造 multimodal content。"""
        content: List[Dict] = []

        if ref_image_paths:
            for i, p in enumerate(ref_image_paths):
                if not p or not os.path.exists(p):
                    logger.warning(f"参考图像不存在，跳过: {p}")
                    continue
                content.append({"type": "text", "text": f"[输入1/4] 参考图像 (Ref Image) 第{i+1}张："})
                content.append({"type": "image", "image": p})
        else:
            content.append({"type": "text", "text": "[输入1/4] 参考图像 (Ref Image)：无"})

        if control_video_path and os.path.exists(control_video_path):
            content.append({"type": "text", "text": "[输入2/4] 控制视频 (Control Video)："})
            content.append({
                "type": "video",
                "video": control_video_path,
                "max_frames": self.video_max_frames,
                "fps": self.video_fps,
            })
        else:
            content.append({"type": "text", "text": "[输入2/4] 控制视频 (Control Video)：无"})

        prompt_text = prompt_text or ""
        content.append({
            "type": "text",
            "text": f"[输入3/4] 文本Prompt (Prompt):\n{prompt_text}"
        })

        if not generated_video_path or not os.path.exists(generated_video_path):
            raise FileNotFoundError(f"生成视频不存在: {generated_video_path}")
        content.append({"type": "text", "text": "[输入4/4] 生成视频 (Generated Video)（待评估）："})
        content.append({
            "type": "video",
            "video": generated_video_path,
            "max_frames": self.video_max_frames,
            "fps": self.video_fps,
        })

        content.append({
            "type": "text",
            "text": "请严格按照上述 System Prompt 中定义的 JSON 格式输出评估结果，不要输出任何其他内容。"
        })
        return content

    def _generate(self, messages: List[Dict]) -> str:
        from vllm import SamplingParams

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as e:
            raise ImportError("缺少 qwen_vl_utils，请安装：pip install qwen-vl-utils") from e

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        mm_data = {}
        if image_inputs:
            mm_data["image"] = image_inputs
        if video_inputs:
            mm_data["video"] = video_inputs

        sampling_params = SamplingParams(temperature=0.0, max_tokens=self.max_new_tokens)

        outputs = self.llm.generate(
            {
                "prompt": text,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs,
            },
            sampling_params=sampling_params,
        )

        if outputs and outputs[0].outputs:
            return outputs[0].outputs[0].text
        return ""

    def evaluate(
        self,
        evaluator_id: str,
        ref_image_paths: Optional[List[str]] = None,
        control_video_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        generated_video_path: Optional[str] = None,
    ) -> Dict:
        if evaluator_id not in EVALUATOR_REGISTRY:
            raise ValueError(f"未知评估器: {evaluator_id}. 可选: {list_evaluators()}")
        if not generated_video_path:
            raise ValueError("generated_video_path 不能为空")

        self.load()

        system_prompt = get_system_prompt(evaluator_id)
        user_content = self._build_user_content(
            ref_image_paths=ref_image_paths,
            control_video_path=control_video_path,
            prompt_text=prompt_text,
            generated_video_path=generated_video_path,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        t0 = time.time()
        raw = self._generate(messages)
        elapsed = time.time() - t0

        parsed = parse_evaluator_output(raw, evaluator_id=evaluator_id)
        parsed["evaluator_id"] = evaluator_id
        parsed["evaluator_name_zh"] = get_evaluator_name_zh(evaluator_id)
        parsed["model_variant"] = self.model_variant
        parsed["model_name_or_path"] = self.model_name_or_path
        parsed["elapsed_sec"] = round(elapsed, 3)
        return parsed


# ============================================================================
# 4. DYNAMIC_EVALUATE —— 批量评估驱动（按 JSONL 中 evaluators 字段动态选评估器）
# ============================================================================


def _pick_generated_video(item: Dict) -> Optional[str]:
    for key in ("pred_lose_path", "lose_path", "pred_video", "path"):
        p = item.get(key)
        if p and isinstance(p, str):
            return p
    return None


def _extract_ref_image_paths(item: Dict) -> List[str]:
    paths = []
    for ref_group in item.get("ref_img_infos", []) or []:
        if isinstance(ref_group, str):
            if ref_group:
                paths.append(ref_group)
        elif isinstance(ref_group, dict):
            for ref_img in ref_group.get("ref_imgs", []) or []:
                p = ref_img.get("img_path") if isinstance(ref_img, dict) else None
                if p and isinstance(p, str):
                    paths.append(p)
    for p in item.get("ref_images", []) or []:
        if isinstance(p, str):
            paths.append(p)
    return paths


def _extract_prompt_text(item: Dict) -> str:
    dense = item.get("dense_caption", []) or []
    if isinstance(dense, list):
        for entry in dense:
            if isinstance(entry, dict):
                c = entry.get("content") or entry.get("dense_caption") or ""
                if c:
                    return c
    elif isinstance(dense, str):
        return dense
    if item.get("prompt_used"):
        return str(item["prompt_used"])
    if item.get("prompt"):
        return str(item["prompt"])
    cap = item.get("caption", []) or []
    if isinstance(cap, list):
        for entry in cap:
            if isinstance(entry, dict):
                c = entry.get("content") or ""
                if c:
                    return c
    elif isinstance(cap, str):
        return cap
    return ""


def _resolve_evaluator_id(name: str) -> Optional[str]:
    """把一个 evaluator 标识（中文名 / 英文 ID / 别名）解析为合法 ID。"""
    if not isinstance(name, str):
        return None
    key = name.strip()
    if not key:
        return None
    if key in EVALUATOR_REGISTRY:
        return key
    if key in _NAME_ZH_TO_ID:
        return _NAME_ZH_TO_ID[key]
    if key in _EVALUATOR_NAME_ALIASES:
        return _EVALUATOR_NAME_ALIASES[key]
    return None


def _extract_evaluators(item: Dict, fallback: Optional[List[str]] = None) -> List[str]:
    """
    抽取要运行的评估器列表（尊重 JSONL 中的 evaluators 字段）。
    兼容英文 ID / 中文名 / 别名。若字段缺失且未提供 fallback，则返回空列表。
    """
    raw = item.get("evaluator")
    if raw is None:
        raw = item.get("evaluators")
    if raw is None:
        return list(fallback) if fallback else []

    if isinstance(raw, list):
        candidates = [x for x in raw if isinstance(x, str)]
    elif isinstance(raw, str):
        candidates = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        return list(fallback) if fallback else []

    resolved, seen, unknown = [], set(), []
    for cand in candidates:
        eid = _resolve_evaluator_id(cand)
        if eid is None:
            unknown.append(cand)
            continue
        if eid not in seen:
            seen.add(eid)
            resolved.append(eid)

    if unknown:
        logger.debug(f"忽略未注册的评估器名: {unknown}")
    if not resolved:
        return list(fallback) if fallback else []
    return resolved


def _sample_id(item: Dict, idx: int) -> str:
    if item.get("sample_id"):
        return str(item["sample_id"])
    gen = _pick_generated_video(item) or ""
    if gen:
        stem = os.path.basename(os.path.dirname(gen)) or os.path.splitext(os.path.basename(gen))[0]
        if stem:
            return f"{idx:05d}_{stem}"
    return f"sample_{idx:05d}"


def detect_file_format(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            if first_char == '[':
                return 'json'
            elif first_char == '{':
                return 'jsonl'
            else:
                f.seek(0)
                content = f.read(1000)
                if content.strip().startswith('['):
                    return 'json'
                elif any(line.strip().startswith('{') for line in content.split('\n') if line.strip()):
                    return 'jsonl'
    except Exception:
        pass
    return 'jsonl'


def load_input_data(path: str) -> List[Dict]:
    file_format = detect_file_format(path)
    if file_format == 'json':
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"JSON 文件应该包含一个数组，但得到的是 {type(data).__name__}")
        return data
    else:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"跳过无效 JSON 行 {line_idx}: {e}")
        return items


def dynamic_evaluate(args) -> None:
    """批量评估主流程（供 CLI 调用）。args 需要包含以下属性：
    input_jsonl, output_jsonl, model_path, model_variant, tensor_parallel_size,
    max_model_len, gpu_memory_utilization, start_idx, end_idx, max_new_tokens,
    video_max_frames, video_fps, dtype, default_evaluators(Optional[List[str]])
    """
    items = load_input_data(args.input_jsonl)
    logger.info(f"加载 {len(items)} 条样本")

    end_idx = args.end_idx if args.end_idx > 0 else len(items)
    items = items[args.start_idx: end_idx]
    logger.info(f"本次评估范围: [{args.start_idx}, {end_idx}), 共 {len(items)} 条")

    total_evaluators = set()
    for item in items:
        total_evaluators.update(_extract_evaluators(item, fallback=args.default_evaluators))
    if not total_evaluators:
        logger.error("所有样本都缺少有效的 evaluators 字段，请检查输入文件")
        sys.exit(1)
    logger.info(f"检测到的评估器类型（{len(total_evaluators)} 种）: {sorted(total_evaluators)}")

    evaluator = VLLMEvaluator(
        model_variant=args.model_variant,
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        video_max_frames=args.video_max_frames,
        video_fps=args.video_fps,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    evaluator.load()

    score_stats = {eid: [] for eid in total_evaluators}
    total_with = 0
    t_total = time.time()

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for idx, item in enumerate(items):
            sample_id = _sample_id(item, idx + args.start_idx)
            evaluators = _extract_evaluators(item, fallback=args.default_evaluators)

            if not evaluators:
                record = {
                    "sample_id": sample_id,
                    "source_index": idx + args.start_idx,
                    "skip_reason": "no_valid_evaluators",
                    "results": {},
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            total_with += 1
            gen_video = _pick_generated_video(item)
            control_video = item.get("control_path", "")
            ref_imgs = _extract_ref_image_paths(item)
            prompt_text = _extract_prompt_text(item)

            logger.info(f"\n{'='*60}")
            logger.info(f"[{idx+1}/{len(items)}] {sample_id}")
            logger.info(f"  evaluators      : {evaluators}")
            logger.info(f"  generated_video : {gen_video}")

            if not gen_video or not os.path.exists(gen_video):
                record = {
                    "sample_id": sample_id,
                    "source_index": idx + args.start_idx,
                    "skip_reason": "generated_video_missing",
                    "generated_video": gen_video,
                    "results": {},
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            per_eval_results = {}
            for eid in evaluators:
                logger.info(f"  >> 评估器 {eid} ({EVALUATOR_REGISTRY[eid]['name_zh']}) ...")
                try:
                    res = evaluator.evaluate(
                        evaluator_id=eid,
                        ref_image_paths=ref_imgs,
                        control_video_path=control_video,
                        prompt_text=prompt_text,
                        generated_video_path=gen_video,
                    )
                except Exception as e:
                    logger.exception(f"     评估失败: {e}")
                    res = {
                        "success": False,
                        "evaluator_id": eid,
                        "score": None,
                        "parse_error": f"exception: {e}",
                        "raw_response": "",
                    }
                per_eval_results[eid] = res
                logger.info(f"     score={res.get('score')}, success={res.get('success')}, elapsed={res.get('elapsed_sec', 'NA')}s")
                if res.get("score") is not None:
                    score_stats[eid].append(float(res["score"]))

            record = {
                "sample_id": sample_id,
                "source_index": idx + args.start_idx,
                "evaluators": evaluators,
                "generated_video": gen_video,
                "control_video": control_video,
                "ref_images": ref_imgs,
                "prompt_text": prompt_text,
                "results": per_eval_results,
            }
            for key, value in item.items():
                if key not in record:
                    record[key] = value

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    logger.info(f"\n{'='*60}")
    logger.info("评估完成汇总：")
    logger.info(f"总样本数: {len(items)}")
    logger.info(f"有评估器的样本数: {total_with}")
    logger.info(f"总耗时: {round(time.time() - t_total, 3)}s")
    if total_with > 0:
        for eid in sorted(total_evaluators):
            scores = score_stats[eid]
            avg = (sum(scores) / len(scores)) if scores else None
            logger.info(f"  {eid:45s} | avg={avg:.3f} | valid={len(scores)}/{total_with}")


# ============================================================================
# 5. SELECT_BEST —— 对 N 份打分结果按 sample_id 对齐，挑出每条最优
# ============================================================================


def load_jsonl(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        if not first:
            return []
        if first.lstrip().startswith("["):
            f.seek(0)
            data = json.load(f)
            return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []
        f.seek(0)
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} 第 {i} 行 JSON 解析失败: {e}")
            if isinstance(obj, dict):
                records.append(obj)
    return records


def extract_scores(record: Dict) -> "OrderedDict[str, Optional[float]]":
    scores = OrderedDict()
    evaluators = record.get("evaluators") or []
    results = record.get("results") or {}
    ordered_keys = []
    for ev in evaluators:
        if ev not in ordered_keys:
            ordered_keys.append(ev)
    for ev in results.keys():
        if ev not in ordered_keys:
            ordered_keys.append(ev)
    for ev in ordered_keys:
        info = results.get(ev) or {}
        score = info.get("score", None) if isinstance(info, dict) else None
        if isinstance(score, bool):
            score = None
        scores[ev] = float(score) if isinstance(score, (int, float)) else None
    return scores


def aggregate(scores: "OrderedDict[str, Optional[float]]") -> Tuple[float, Optional[float], int, int]:
    total = 0.0
    valid = 0
    for v in scores.values():
        if v is not None:
            total += v
            valid += 1
    mean = (total / valid) if valid > 0 else None
    return total, mean, valid, len(scores)


def source_label(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def select_best(input_paths: List[str], rank_by: str = "total") -> Dict:
    assert rank_by in ("total", "mean"), "rank_by 只能是 total 或 mean"
    sources = [load_jsonl(p) for p in input_paths]
    labels = [source_label(p) for p in input_paths]

    seen: "OrderedDict[str, None]" = OrderedDict()
    for recs in sources:
        for r in recs:
            sid = r.get("sample_id")
            if sid is not None and sid not in seen:
                seen[sid] = None
    sample_ids = list(seen.keys())

    indexed = []
    for recs in sources:
        idx = {}
        for r in recs:
            sid = r.get("sample_id")
            if sid is not None:
                idx[sid] = r
        indexed.append(idx)

    samples_out = []
    miss_counter = 0
    for sid in sample_ids:
        per_source = []
        for label, idx in zip(labels, indexed):
            rec = idx.get(sid)
            if rec is None:
                miss_counter += 1
                per_source.append({
                    "source": label, "present": False, "total_score": None,
                    "mean_score": None, "num_evaluators": 0, "valid_evaluators": 0,
                    "per_evaluator": {}, "generated_video": None,
                })
                continue
            scores = extract_scores(rec)
            total, mean, valid, n = aggregate(scores)
            per_source.append({
                "source": label, "present": True,
                "total_score": round(total, 4),
                "mean_score": round(mean, 4) if mean is not None else None,
                "num_evaluators": n, "valid_evaluators": valid,
                "per_evaluator": {k: v for k, v in scores.items()},
                "generated_video": rec.get("generated_video"),
            })

        def _key(item):
            if not item["present"]:
                return (-1.0, -1)
            v = item["mean_score"] if rank_by == "mean" else item["total_score"]
            if v is None:
                return (-1.0, -1)
            return (v, item["valid_evaluators"])

        best_idx = max(range(len(per_source)), key=lambda i: _key(per_source[i]))
        best = per_source[best_idx]
        samples_out.append({
            "sample_id": sid,
            "per_source": per_source,
            "best_index": best_idx,
            "best_source": best["source"],
            "best_total_score": best["total_score"],
            "best_mean_score": best["mean_score"],
            "best_generated_video": best["generated_video"],
            "rank_by": rank_by,
        })

    source_summary = []
    for i, label in enumerate(labels):
        totals = [s["per_source"][i]["total_score"] for s in samples_out
                  if s["per_source"][i]["present"] and s["per_source"][i]["total_score"] is not None]
        means = [s["per_source"][i]["mean_score"] for s in samples_out
                 if s["per_source"][i]["present"] and s["per_source"][i]["mean_score"] is not None]
        win_cnt = sum(1 for s in samples_out if s["best_index"] == i)
        source_summary.append({
            "source": label,
            "num_present": len(totals),
            "avg_total_score": round(sum(totals) / len(totals), 4) if totals else None,
            "avg_mean_score": round(sum(means) / len(means), 4) if means else None,
            "win_count": win_cnt,
        })

    return {
        "meta": {
            "source_files": [os.path.abspath(p) for p in input_paths],
            "source_labels": labels,
            "num_sources": len(input_paths),
            "num_samples": len(sample_ids),
            "num_missing_records": miss_counter,
            "rank_by": rank_by,
        },
        "source_summary": source_summary,
        "samples": samples_out,
    }


# ============================================================================
# 6. CLI —— 支持 evaluate / select-best 两个子命令
# ============================================================================

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CogOmniControl —— 基于 Qwen3-VL-30B (vLLM) 的视频生成评估与择优",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ---- evaluate ----
    pe = sub.add_parser("evaluate", help="用评估器对一份候选 JSONL 打分")
    pe.add_argument("--input_jsonl", required=True, help="输入 jsonl（需含 evaluators / generated_video 等字段）")
    pe.add_argument("--output_jsonl", required=True, help="输出打分结果 jsonl")
    pe.add_argument("--model_path", default=None, help="模型路径，如 Qwen/Qwen3-VL-30B-A3B-Thinking-FP8")
    pe.add_argument("--model_variant", default="thinking", choices=list(VLLM_MODEL_VARIANT_MAP.keys()))
    pe.add_argument("--tensor_parallel_size", type=int, default=2)
    pe.add_argument("--max_model_len", type=int, default=32768)
    pe.add_argument("--gpu_memory_utilization", type=float, default=0.65)
    pe.add_argument("--max_new_tokens", type=int, default=4096)
    pe.add_argument("--video_max_frames", type=int, default=32)
    pe.add_argument("--video_fps", type=float, default=2.0)
    pe.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"])
    pe.add_argument("--start_idx", type=int, default=0)
    pe.add_argument("--end_idx", type=int, default=-1)
    # 当输入 JSONL 没有 evaluators 字段时，用此默认列表（默认跑全部 11 个）
    pe.add_argument("--default_evaluators", nargs="*", default=None,
                   help="输入缺少 evaluators 字段时使用的默认评估器（默认全部 11 个）")
    pe.add_argument("--list_evaluators", action="store_true", help="仅打印可用评估器并退出")

    # ---- select-best ----
    pb = sub.add_parser("select-best", help="对 N 份打分结果挑出每条最优")
    pb.add_argument("--inputs", nargs="+", required=True, help="N 份打分 jsonl 文件路径")
    pb.add_argument("--output", required=True, help="输出 json 文件")
    pb.add_argument("--rank-by", choices=["total", "mean"], default="total",
                    help="按 total（总分）或 mean（平均分）排名")

    return p


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    args = _build_argparser().parse_args()

    if args.command == "evaluate":
        if args.list_evaluators:
            print("可用评估器：")
            for eid, info in EVALUATOR_REGISTRY.items():
                print(f"  - {eid:45s} | {info['name_zh']}")
            return
        if args.default_evaluators is None:
            args.default_evaluators = ALL_EVALUATORS
        dynamic_evaluate(args)
    elif args.command == "select-best":
        result = select_best(args.inputs, rank_by=args.rank_by)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[DONE] 已输出 {result['meta']['num_samples']} 条样本 -> {args.output}")
        print("[SUMMARY] 各来源胜出统计:")
        for s in result["source_summary"]:
            print(f"   - {s['source']}: wins={s['win_count']}, "
                  f"avg_total={s['avg_total_score']}, avg_mean={s['avg_mean_score']}, "
                  f"present={s['num_present']}")


if __name__ == "__main__":
    main()
