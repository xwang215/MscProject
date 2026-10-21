"""Build factual bilingual reports from saved metrics."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'results').glob('three_class_*.json'))]
    audit = json.loads((ROOT/'data/processed/audit.json').read_text(encoding='utf-8'))
    headers = '| Model | Accuracy | Balanced accuracy | Macro-F1 | QWK | MAE | Macro-F1 95% CI |\n|---|---:|---:|---:|---:|---:|---|'
    table = headers
    for r in results:
        lo, hi = r['macro_f1_image_bootstrap_95ci']
        table += f"\n| {r['model']} | {r['accuracy']:.4f} | {r['balanced_accuracy']:.4f} | {r['macro_f1']:.4f} | {r['quadratic_weighted_kappa']:.4f} | {r['ordinal_index_mae']:.4f} | [{lo:.4f}, {hi:.4f}] |"
    if len(results) != 3:
        raise RuntimeError('All three actual experiments are required to generate a completed report.')
    if len({r['manifest_sha256'] for r in results}) != 1:
        raise RuntimeError('Models were not evaluated on the same manifest.')
    best = max(results, key=lambda r: r['macro_f1'])
    cnn = next(r for r in results if r['model'] == 'resnet18')
    differences = ', '.join(f"{r['model']}: {r['macro_f1']-cnn['macro_f1']:+.4f}" for r in results if r != cnn)
    severe_recalls = ', '.join(f"{r['model']}: {r['classification_report']['3']['recall']:.4f}" for r in results)
    counts = results[0]['counts']
    with (ROOT/'data/processed/three_class.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    from collections import Counter
    train_counts = Counter(r['grade'] for r in rows if r['split'] == 'train')
    majority = train_counts.most_common(1)[0][0]
    test_rows = [r for r in rows if r['split'] == 'test']
    majority_acc = sum(r['grade'] == majority for r in test_rows)/len(test_rows)
    details = '\n\n'.join(f"### {r['model']}\n\nSelected C: {r['selected_C']}; validation macro-F1: {r['validation_macro_f1']:.4f}; elapsed: {r['elapsed_seconds']:.1f} s.\n\nConfusion matrix (rows=true, columns=predicted; order 1, 2, 3):\n\n```text\n"+'\n'.join(str(row) for row in r['confusion_matrix'])+'\n```' for r in results)
    details_zh = '\n\n'.join(f"### {r['model']}\n\n选定 C：{r['selected_C']}；验证集宏平均 F1：{r['validation_macro_f1']:.4f}；耗时：{r['elapsed_seconds']:.1f} 秒。\n\n混淆矩阵（行是真实等级，列是预测等级；顺序为 1、2、3）：\n\n```text\n"+'\n'.join(str(row) for row in r['confusion_matrix'])+'\n```' for r in results)
    sources = '''1. Borgli et al. (2020). HyperKvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy. Scientific Data 7, 283. https://doi.org/10.1038/s41597-020-00622-y
2. Official dataset: https://datasets.simula.no/hyper-kvasir/ ; repository and terms: https://github.com/simula/hyper-kvasir
3. ResNet-18 documentation: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
4. DeiT-Tiny model card: https://huggingface.co/facebook/deit-tiny-patch16-224
5. Swin-Tiny model card: https://huggingface.co/microsoft/swin-tiny-patch4-window7-224
6. Kvasir-VQA: https://datasets.simula.no/kvasir-vqa/ ; https://huggingface.co/datasets/SimulaMet-HOST/Kvasir-VQA
7. UC grading study: https://pmc.ncbi.nlm.nih.gov/articles/PMC8854553/'''
    en = f'''# Ulcerative Colitis Severity Estimation: Reproducible Baseline Study

Project date: 20 October 2026 (America/Toronto). This report describes executed experiments, not a clinical validation study.

## Objective and dataset selection

The objective was to compare a CNN with two Hugging Face vision Transformers on UC endoscopic severity estimation. HyperKvasir's already labeled images were selected because their UC subset contains severity labels. Ordinary disease-category classification would not answer this severity question. The original six labels were retained. Definite grades 1, 2, and 3 define the primary three-class task; interval grades are reserved for a separate six-class study. The primary task excludes ambiguous cases and does not cover grade 0. It should not be described as a complete four-grade Mayo classifier. [1, 2, 7]

## Completed work and data preparation

The official labeled archive was downloaded locally. Scripts were written for resumable download, archive CRC verification, SHA-256 provenance, UC extraction, image validation, duplicate detection, grouped splitting, model comparison, and report generation. English code comments are accompanied by Simplified Chinese translations. Raw images and pretrained weights are excluded from Git; code, reports, manifests, audit metadata, and actual predictions are suitable for version control.

Observed archive class counts: {audit['archive_counts']}. Readable UC images: {audit['readable']}; exact duplicate copies excluded: {audit['duplicates']}; conflicting duplicate images excluded: {audit['conflicting']}; retained images: {audit['retained']}; flagged perceptually similar pairs: {audit['similar_pairs']}. Decode errors: {len(audit['errors'])}.

Images remain unchanged on disk. RGB conversion and model-specific resizing, cropping, and normalization occur at model input. Exact pixel duplicates are collapsed. Pairs with 64-bit difference-hash distance at most four are connected into groups kept in one split. The grouping is a heuristic requiring review; it neither detects all near-duplicates nor establishes patient identity. Five stratified group folds define approximately 60% training, 20% validation, and 20% testing with seed 42. Actual primary-task counts: {counts}.

Visual spot-checking found black borders, text overlays, and green navigation insets in sample images. These may act as confounders. They were not automatically removed in this baseline. A prespecified masking/cropping sensitivity analysis is needed before attributing model decisions to mucosal findings.

## Experimental method

ResNet-18, DeiT-Tiny, and Swin-Tiny use pretrained ImageNet representations. Each encoder is frozen in evaluation mode. ResNet uses its pre-classifier features, DeiT its final class token, and Swin its mean final token features. A standardized, class-balanced logistic regression classifier is fitted separately for each encoder. Standardization uses training data only. C is selected from 0.01, 0.1, 1, and 10 using validation macro-F1, followed by a single evaluation on the shared test manifest. Model-specific preprocessing is retained, so the comparison concerns complete pretrained pipelines rather than architecture alone. No encoder fine-tuning or training augmentation was performed. [3–5]

The recorded hardware device was {results[0]['device']}. Exact installed packages are recorded in requirements-lock.txt. Model identifiers/revisions, manifest hashes, predictions, class-wise metrics, and timings are recorded under results/. This is a practical CPU baseline, not a comparison of fully optimized end-to-end networks.

## Actual test results

{table}

The training-majority classifier always predicts grade {majority}; its accuracy on this test set is {majority_acc:.4f}. Accuracy alone is insufficient because the classes are imbalanced. Macro-F1 and balanced accuracy give minority classes greater visibility. QWK and MAE describe ordered errors for this three-grade task. The displayed intervals use 1,000 image-level bootstrap resamples; they are descriptive and do not account for unknown patient clustering.

{details}

## Comparison and conclusions

The highest observed test macro-F1 was achieved by {best['model']} ({best['macro_f1']:.4f}). Transformer macro-F1 differences relative to ResNet-18 were {differences}. These numbers describe this split and this frozen-feature protocol. They do not establish a statistically significant advantage or prove that one architecture is generally superior. Hyperparameter selection used validation results; the test ranking is reported descriptively and must not drive further tuning on this same test set.

Grade-3 recall was {severe_recalls}. Thus the strongest aggregate performer was not the most sensitive to severe cases. The choice of primary metric should reflect a prespecified research objective; a clinical success threshold was not defined here. Timings are not a controlled speed benchmark because thread counts differed and processes ran concurrently.

The completed work provides a reproducible starting point and a real comparison. Stronger conclusions require expert review of similar-image groups and errors, multiple predefined seeds or grouped cross-validation, matched fine-tuning budgets, and an external patient-independent test set. There are no patient identifiers available in this workflow, so patient-independent generalization cannot be claimed. Image-based endoscopic grades do not represent the full clinical severity of UC.

## VQA decision and next stage

No success threshold was supplied, so these results cannot objectively be declared clinically adequate. VQA was not trained in this stage. Lower classification performance alone does not show that VQA will improve severity prediction. Kvasir-VQA is an optional extension containing image-question-answer annotations; its general abnormalities, color, and location questions must not be assumed to provide expert Mayo severity labels. Before using it, audit severity supervision and overlap with the locked HyperKvasir evaluation images. Group all questions for each image together and prevent target-answer leakage. If an expert-validated severity mapping exists, compare a VQA-assisted method against these baselines on a prespecified independent evaluation set. [6]

## Reproduction and limitations

Follow README.md from the project root. Run download_data.py, prepare_data.py, the three compare_models.py commands, and write_reports.py. The six-class manifest is prepared but its models have not been evaluated in this report. The study uses a small, imbalanced, single public dataset, excludes ambiguous grades from its primary task, retains model-specific preprocessing, and lacks patient identifiers and clinical validation. These limitations apply even if observed accuracy is high.

## References

{sources}
'''
    zh = f'''# 溃疡性结肠炎严重程度估计：可复现基线研究

项目日期：2026 年 10 月 20 日（多伦多时区）。本报告描述实际执行的实验，不属于临床验证研究。

## 研究目标与数据集选择

本研究比较一个 CNN 与两个 Hugging Face 视觉 Transformer 的 UC 内镜严重程度估计能力。选择 HyperKvasir 已标注图像，是因为其中的 UC 子集具有严重程度标签。普通疾病类别识别不能回答严重程度问题。保留全部六种原始标签；明确的 1、2、3 级构成主要三分类任务，区间等级保留给独立的六分类研究。主任务排除模糊病例且不覆盖 0 级，不能称为完整的四级 Mayo 分类器。[1、2、7]

## 已完成工作与数据处理

已将官方已标注压缩包下载到本地，并编写支持断点续传、压缩包 CRC 验证、SHA-256 来源记录、UC 提取、图像验证、重复检测、分组划分、模型对比及报告生成的脚本。代码英文注释附有简体中文翻译。原始图像和预训练权重不纳入 Git；代码、报告、清单、审核元数据及实际预测可用于版本控制。

压缩包实际类别数量：{audit['archive_counts']}。可读取 UC 图像：{audit['readable']}；排除的完全重复副本：{audit['duplicates']}；排除的标签冲突重复图像：{audit['conflicting']}；保留图像：{audit['retained']}；标记的感知相似图像对：{audit['similar_pairs']}。解码错误：{len(audit['errors'])}。

磁盘中的图像保持原样。RGB 转换及模型配套的缩放、裁剪、标准化在输入模型时进行。像素完全相同的副本合并为一份。64 位差分哈希距离不超过四的图像对被连接成组，同组留在同一划分中。这是需要复核的启发式方法，既不能发现所有近似重复，也不能确认患者身份。以随机种子 42 创建五折分层分组划分，近似得到 60% 训练、20% 验证和 20% 测试。主任务实际数量：{counts}。

抽样目视检查发现部分图像存在黑色边框、文字叠加和绿色导航小窗，这些可能成为混杂因素。本基线未自动去除这些元素。在把模型判断归因于黏膜表现之前，需要预先设计遮挡／裁剪敏感性分析。

## 实验方法

ResNet-18、DeiT-Tiny 和 Swin-Tiny 使用 ImageNet 预训练表征。编码器固定为评估模式并冻结参数。ResNet 使用分类头之前的特征，DeiT 使用最终分类 token，Swin 使用最终 token 的平均特征。每种编码器分别拟合带特征标准化和类别平衡权重的逻辑回归分类器。标准化只拟合训练数据，通过验证集宏平均 F1 从 0.01、0.1、1、10 中选择 C，然后在共享测试清单上评估一次。保留各模型原有预处理，因此比较的是完整预训练流程，而不单是网络结构。未进行编码器微调或训练数据增强。[3–5]

记录的计算设备为 {results[0]['device']}。精确安装版本记录于 requirements-lock.txt。模型标识与版本、清单哈希、预测、各类别指标及耗时记录在 results/ 中。这是可在 CPU 上运行的实用基线，不是充分优化的端到端网络对比。

## 实际测试结果

表头含义：Model 为模型；Accuracy 为准确率；Balanced accuracy 为平衡准确率；Macro-F1 为宏平均 F1；QWK 为二次加权 Kappa；MAE 为平均绝对误差；CI 为置信区间。

{table}

训练集多数类分类器始终预测 {majority} 级，在本测试集的准确率为 {majority_acc:.4f}。类别不平衡使准确率本身不够充分。宏平均 F1 与平衡准确率更能反映少数类别表现。QWK 和 MAE 描述三等级任务的有序错误。表中区间使用 1,000 次图像级自助抽样，仅作描述，未考虑未知患者聚类。

{details_zh}

## 对比与结论

观察到的最高测试集宏平均 F1 来自 {best['model']}（{best['macro_f1']:.4f}）。Transformer 相对 ResNet-18 的宏平均 F1 差值为 {differences}。这些数字仅描述本次划分及冻结特征方案，不能证明具有统计显著优势，也不能证明某种结构普遍更好。超参数使用验证集选择；测试集排名仅作描述，不得用于继续调整同一测试集上的模型。

3 级召回率为 {severe_recalls}。因此总体指标最好的模型并非对重度病例最敏感的模型。主要评价指标应由预先规定的研究目标决定，本研究没有定义临床成功阈值。由于线程数不同且进程并发运行，记录的耗时不能作为受控速度基准。

已完成的工作提供了可复现的起点及真实对比。更强的结论需要专家复核相似图像组与错误、多组预先规定的随机种子或分组交叉验证、匹配的微调预算，以及外部患者独立测试集。本流程没有可用的患者编号，因此不能宣称患者独立的泛化能力。基于图像的内镜分级也不代表 UC 完整的临床严重程度。

## VQA 决策与后续阶段

目前未提供成功阈值，因此无法客观判断这些结果是否满足临床要求。本阶段未训练 VQA。分类表现有限本身不能证明 VQA 会改善严重程度预测。Kvasir-VQA 是包含图像问答标注的可选扩展；不能假定其一般异常、颜色和位置问题提供了专家 Mayo 严重程度标签。使用前必须审核严重程度监督，并检查与已锁定的 HyperKvasir 评估图像是否重叠。同一图像的所有问题必须分在一起，防止目标答案泄漏。如果存在专家验证的严重程度映射，才应在预先规定的独立评估集上比较 VQA 辅助方法与当前基线。[6]

## 复现与局限

从项目根目录按 README.md 操作：运行 download_data.py、prepare_data.py、三次 compare_models.py 和 write_reports.py。六分类清单已准备，但本报告未评估六分类模型。研究采用单个公开的小型不平衡数据集，主任务排除模糊等级，保留模型专用预处理，且缺少患者编号和临床验证。即使观察到较高准确率，这些局限仍然成立。

## 参考文献

{sources}
'''
    (ROOT/'reports/report_en.md').write_text(en, encoding='utf-8')
    (ROOT/'reports/report_zh.md').write_text(zh, encoding='utf-8')
    print('Wrote both reports from actual metrics.')


if __name__ == '__main__':
    main()
