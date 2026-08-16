# TopoContour-Mamba
A topology-guided Vision Mamba that scans visual features along contour normals, contour paths, and contour split–merge structures.
# 基于等高线拓扑的 Mamba 视觉扫描

## 项目简介

本项目尝试让图像自身的空间结构决定 Mamba 的扫描顺序。图像先被转换为结构强度图，再提取多个层级的等高线。等高线层级表示结构变化的顺序；同一层级中互不相连的等高线没有天然先后关系，因此保持并行。

模型主要处理三种空间关系：沿法线读取相邻等高线之间的变化，沿轮廓整合同一条等高线的信息，再沿等高线的延续、分裂和汇合关系传播全局状态。热力图和等高线可以先由现有方法生成，本项目重点研究后续的采样、汇集和 Mamba 状态传播。

## 方法流程

1. **提取分层等高线。** 从结构强度图中获得多个层级的等高线，并保持层级间隔一致，使轮廓间距能够反映局部变化强弱。

2. **在等高线上稀疏采样。** 按照真实弧长选择采样点，并从浅层特征图中读取颜色、纹理和边缘特征。结构变化明显的位置可以采得更密，平滑位置可以采得更稀，从而减少无意义的视觉 Token。

3. **沿法线连接相邻等高线。** 从当前采样点沿法线向两侧延伸，直到分别到达上下相邻层级的等高线。法线段长度由真实轮廓间距决定，不使用固定大小的局部窗口。

4. **把层间信息汇聚到当前点。** 两侧特征分别从相邻等高线向当前采样点进行 Mamba 扫描，并在当前点融合。这样既能读取轮廓内外的视觉信息，也能记录不同层级之间的变化过程；等高线越密，通常表示该区域变化越快。

5. **沿等高线进行双向扫描。** 将同一条等高线上的采样点按照真实轮廓顺序排列，分别沿两个方向传播。对于闭合轮廓，可以在结构变化明显的位置设置汇集点，两个方向最终在该位置融合，形成整条等高线的状态。

6. **按照拓扑关系传播状态。** 同一层级中的独立等高线并行处理，不强行排序；不同层级之间则按照轮廓的延续、分裂和汇合关系传递状态。

7. **完成图像分类。** 汇聚各条等高线经过拓扑传播后的状态，再送入分类头得到结果。

## 分支状态传播

等高线随着层级变化会产生不同的拓扑关系，模型分别处理：

1. **延续：** 一条等高线演化为下一层的一条等高线时，上一条轮廓的状态传给下一条轮廓，并与新轮廓自身的扫描结果融合。

2. **分裂：** 一条父轮廓分裂成多个子轮廓时，父状态分别传入各个分支，每个子轮廓再结合自己的特征独立更新。不同分支保留共同来源，但不会得到完全相同的状态。

3. **汇合：** 多条轮廓汇合时，模型根据各分支与新轮廓的关系，对多个父状态进行加权融合，再交给汇合后的轮廓继续传播。

分裂不是简单复制，汇合也不是直接求和，而是由可训练的门控决定各部分信息的保留比例。对于分类任务，还可以分别从低层级向高层级、从高层级向低层级传播，最后融合两个方向的结果。

## 整体结构


1. 图像特征与分层等高线
                
2. 等高线稀疏采样
                
3. 相邻等高线之间的法线扫描
                
4. 等高线双向扫描与汇集
                
5. 延续、分裂和汇合状态传播
                
6. 全局汇聚与图像分类


法线 Mamba 负责相邻层级之间的变化，等高线 Mamba 负责同层轮廓内部的关系，拓扑传播模块负责全局结构。扫描方向由图像内容决定，而不是固定的横向、纵向或蛇形路径。

## 后续扩展

- **可训练的结构图：** 使用卷积模块生成与输入尺寸对应的结构图，再利用软化的层级表示连接后续采样模块，使分类损失能够回传到结构图生成网络。早期版本也可以固定这部分，只训练特征提取、Mamba、分支门控和分类头。

- **主方向环扫描：** 以等高线采样点为中心建立多层同心环，用局部主方向统一每个环的起止和汇集位置，先沿环进行双向 Mamba 扫描，再按照从外到内的顺序把各环状态汇聚到中心，并可通过只在关键点启用或降低外环采样密度来减少重叠开销。

- **层间稠密扫描：** 在相邻等高线之间加入横向和纵向扫描，补充稀疏法线没有读取到的区域。稠密位置得到的状态先汇聚到附近的等高线采样点，再进入轮廓扫描和拓扑传播。

- **自适应采样：** 根据轮廓弯曲程度、特征变化和等高线密度调整采样点数量。复杂区域保留更多 Token，平滑区域减少计算。

# TopoContour-Mamba

## Overview

TopoContour-Mamba explores how the spatial structure of an image can guide the scan order of Mamba. An image is first converted into a structural intensity map, from which contours are extracted at multiple levels. The contour levels define the order of structural change, while disconnected contours at the same level remain parallel because they have no natural temporal order.

The model focuses on three spatial relationships: normal-direction scans capture changes between neighboring contours, contour scans integrate information along the same contour, and topology-guided propagation carries states through contour continuation, splitting, and merging. Existing methods can be used to generate the structural map and contours in the initial version; the main focus is the subsequent sampling, aggregation, and Mamba state propagation.

## Method

1. **Extract multi-level contours.** Obtain contours at multiple uniformly spaced levels so that the distance between neighboring contours reflects the strength of local structural variation.

2. **Sample points sparsely along each contour.** Select points according to true arc length and read color, texture, and edge features from a shallow feature map. More points can be retained in complex regions, while smooth regions use fewer visual tokens.

3. **Connect neighboring contours along the normal direction.** From each contour point, extend in both normal directions until reaching the nearest contours at the adjacent upper and lower levels. The normal segment therefore follows the actual spacing between contours instead of using a fixed local window.

4. **Aggregate inter-level information at the contour point.** Features on both sides are scanned from the neighboring contours toward the current point and fused there. This captures visual information inside and outside the contour together with the transition between contour levels; densely packed contours usually indicate faster local variation.

5. **Scan bidirectionally along each contour.** Arrange sampled points in their true contour order and propagate information in both directions. For a closed contour, a structurally informative point can be selected as the aggregation point, where the two directional states are fused into a contour representation.

6. **Propagate states through contour topology.** Independent contours at the same level are processed in parallel, while states across levels follow actual continuation, split, and merge relationships.

7. **Perform image classification.** Aggregate the topology-aware contour states and pass the resulting representation to a classification head.

## Branch-State Propagation

Contour topology changes across levels in three main ways:

1. **Continuation:** When one contour evolves into another contour at the next level, its state is passed forward and fused with the new contour's own scan result.

2. **Split:** When one parent contour divides into several child contours, the parent state is passed to every branch. Each child then updates it using its own features, preserving their shared origin while allowing different representations.

3. **Merge:** When several contours merge, their parent states are weighted according to their relationships with the new contour and then fused before propagation continues.

Splitting is not treated as simple state copying, and merging is not handled by direct summation. Trainable gates determine how much information should be preserved from each branch. For classification, topology propagation may also be performed in both low-to-high and high-to-low level directions before the two results are fused.

## Architecture

```text
1. Image features and multi-level contours
                    ↓
2. Sparse contour sampling
                    ↓
3. Normal scans between neighboring contours
                    ↓
4. Bidirectional contour scanning and aggregation
                    ↓
5. Continuation, split, and merge state propagation
                    ↓
6. Global aggregation and image classification
```

The normal Mamba models changes between neighboring levels, the contour Mamba models relationships within each contour, and the topology module models the global structure. The scan order is determined by image content rather than fixed horizontal, vertical, or snake-like paths.

## Future Extensions

- **Trainable structural map:** A convolutional module can generate a same-resolution structural map, while softened level representations connect it to the sampling stage so that the classification loss can propagate back to the map generator. The initial version can keep this stage fixed and train only the feature extractor, Mamba modules, branch gates, and classifier.

- **Principal-direction ring scanning:** Concentric rings are built around a contour sample point, a local principal direction aligns the start, end, and aggregation position of every ring, bidirectional Mamba scans are first performed along each ring, and the resulting states are then propagated from the outer rings toward the center, with ring scans activated only at key points or sampled more sparsely on outer rings to reduce overlap.

- **Dense inter-contour scanning:** Horizontal and vertical scans can be added inside the regions between neighboring contours to capture information missed by sparse normal segments. The dense states are first aggregated into nearby contour points before entering contour scanning and topology propagation.

- **Adaptive sampling:** The number of contour samples can be adjusted according to curvature, feature variation, and contour density. Complex regions retain more tokens, while smooth regions use less computation.

