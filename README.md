# TopoContour-Mamba

A topology-guided Vision Mamba that gathers features along contour normals, scans closed contours in parallel, propagates states through nested contour branches, and aggregates peak states by intensity.


## TopoContour-Mamba

### 项目简介

TopoContour-Mamba 尝试让图像自身的空间结构决定 Mamba 的扫描顺序。图像先被转换为结构强度图，再按照统一的层级间隔提取闭合等高线。等高线的层级和包含关系形成从外层低强度区域到内层高强度峰值的拓扑结构。

模型先沿法线把外侧区域的信息汇聚到等高线采样点，再让所有闭合等高线独立进行双向 Mamba 扫描。每个圈产生一个局部状态后，拓扑 Mamba 沿包含关系从低层到高层递归汇集；不同峰值产生的最终状态再按照峰值强度从低到高扫描，形成用于分类的全局状态。

### 方法流程

1. **生成分层等高线。** 从结构强度图中提取多个等间隔层级的闭合等高线，并根据空间包含关系构建等高线森林。外层圈是父节点，内部更高层的圈是子节点；图像边框作为最外层轮廓的虚拟边界。

2. **在等高线上稀疏采样。** 按照真实弧长选择采样点，并从浅层特征图中读取颜色、纹理和边缘特征。轮廓弯曲越明显、当前圈与外侧相邻等高线的距离越小，采样越密；平滑且层间距离较大的区域采样更稀。

3. **沿法线汇聚外侧信息。** 从当前采样点沿直线法线找到外侧最近的等高线，再从外侧等高线向当前点运行法线 Mamba。最终状态作为当前采样点的 Token。最外层圈没有外侧等高线时，从虚拟图像边界向当前点扫描。所有法线任务共享参数并可并行执行。

4. **并行扫描所有闭合等高线。** 每个圈选择一个汇集点，默认使用与外侧相邻等高线距离最小的位置；如果无法稳定确定，则使用曲率较大的位置。顺时针和逆时针 Mamba 从同一汇集点出发，分别绕完整个圈后回到该点。两个方向读取顺序相反，在扫描过程中不交换隐藏状态，只在汇集点融合最终结果，产生该圈的局部状态 \(h_C\)。所有圈先独立完成扫描，因此不同圈可以并行处理。

5. **沿等高线森林递归汇集。** 所有圈得到局部状态后，拓扑 Mamba 从最低层外圈开始，按照包含关系向内部高层传播。每个节点结合父历史状态和自己的 \(h_C\) 进行更新；如果一个父圈包含多个子圈，所有子圈继承同一个父历史状态，再根据各自的局部状态独立更新。递归继续到各个叶子节点，每个热力图峰值最终得到一个分支状态 \(h_{peak}\)。

6. **按照峰值强度完成分类。** 只保留各个叶子节点的 \(h_{peak}\)，并使用叶子区域内部的最高热力图响应表示峰值强度。将所有峰值状态从低强度到高强度排列，再交给最终 Mamba 扫描。最后一个隐藏状态作为整张图像的表示，并送入分类头。

### 并行方式

TopoContour-Mamba 将局部视觉扫描和全局拓扑汇集分开处理：

1. 所有法线任务并行产生采样点 Token；
2. 所有闭合等高线独立并行产生局部状态；
3. 拓扑 Mamba 按树的深度从低层到高层传播，同一父节点下的子分支并行更新；
4. 少量峰值状态按照强度顺序完成最终扫描。

顺时针和逆时针扫描可以共享模型参数，但维护相互独立的隐藏状态。父历史状态只用于拓扑汇集，不进入前面的独立轮廓扫描，因此不会阻止所有圈并行编码。

### 整体结构

```text
1. 结构强度图与闭合等高线森林
                  ↓
2. 外侧等高线 → 当前采样点的法线扫描
                  ↓
3. 所有圈独立进行完整的双向闭环扫描
                  ↓
4. 每个圈产生局部状态 h_C
                  ↓
5. 沿包含关系从低层递归到各个峰值
                  ↓
6. 每个峰值产生分支状态 h_peak
                  ↓
7. 按峰值强度从低到高扫描
                  ↓
8. 最终状态 h → 图像分类
```

### 后续扩展

- **可训练的结构图：** 使用卷积模块生成与输入尺寸对应的结构强度图，再利用软化的层级表示连接后续采样，使分类损失能够回传到结构图生成网络。

- **主方向环扫描：** 以等高线采样点为中心建立多层同心环，用局部主方向统一每个环的起止和汇集位置，先沿环进行双向 Mamba 扫描，再按照从外到内的顺序把各环状态汇聚到中心。

- **层间稠密扫描：** 在相邻等高线之间加入横向和纵向扫描，补充稀疏法线没有读取到的区域，再把稠密状态汇聚到附近的等高线采样点。

- **可训练采样策略：** 将当前由曲率和等高线距离控制的采样密度替换为可训练策略，使模型根据任务动态选择采样位置。

- **树形并行扫描：** 进一步研究拓扑 Mamba 的树形并行算法，减少不同等高线深度之间的顺序等待。

## TopoContour-Mamba

### Overview

TopoContour-Mamba lets the spatial structure of each image determine the scan order of Mamba. An image is first converted into a structural intensity map, from which closed contours are extracted at uniformly spaced levels. Their levels and containment relationships form a topology from low-intensity outer regions to high-intensity inner peaks.

The model first gathers outside information into contour points through normal-direction scans and then encodes every closed contour independently with bidirectional Mamba. After each contour produces a local state, a topology Mamba recursively aggregates these states from lower to higher levels. The final states of different peaks are ordered from low to high peak intensity and scanned again to form the global representation for classification.

### Method

1. **Build a multi-level contour forest.** Extract closed contours at uniformly spaced levels and construct their containment relationships. An outer contour becomes the parent of the higher-level contours inside it. The image border is treated as a virtual boundary for the outermost contours.

2. **Sample each contour sparsely.** Select points according to true arc length and read color, texture, and edge features from a shallow feature map. Sampling becomes denser where curvature is higher or the distance to the outside neighboring contour is smaller, while smooth and widely spaced regions use fewer tokens.

3. **Gather outside information along contour normals.** From each contour point, follow its straight normal to the nearest outside contour, and run a normal Mamba from that outer contour toward the current point. The final state becomes the token of the current contour point. For an outermost contour, the scan starts from the virtual image boundary. All normal scans share parameters and can run in parallel.

4. **Scan all closed contours independently.** Each contour selects an aggregation point, by default where its distance to the outside neighboring contour is smallest; a high-curvature point is used when this position cannot be determined reliably. Clockwise and counterclockwise Mamba streams start from the same point, traverse the complete contour, and return to that point. The two streams process opposite orders without exchanging hidden states and fuse only at the aggregation point, producing a local contour state \(h_C\). Since every contour is encoded independently, all contour scans can run in parallel.

5. **Aggregate states recursively through the contour forest.** After all local contour states are available, a topology Mamba starts from the lowest-level outer contours and follows the containment structure toward higher levels. Each node updates the inherited parent history with its own \(h_C\). If one parent contains several child contours, every child receives the same parent history and then evolves independently using its local state. The recursion ends at the leaves, producing one branch state \(h_{peak}\) for each heatmap peak.

6. **Aggregate peaks for classification.** Keep only the leaf states \(h_{peak}\), and define peak intensity using the highest heatmap response inside each leaf region. Sort the peak states from low to high intensity and process them with a final Mamba. Its last hidden state represents the image and is passed to the classification head.

### Parallel Execution

TopoContour-Mamba separates local visual encoding from global topology aggregation:

1. All normal scans generate contour-point tokens in parallel;
2. All closed contours independently generate local states in parallel;
3. The topology Mamba propagates from lower to higher tree depths, while sibling branches update in parallel;
4. A short sequence of peak states is scanned in intensity order.

The clockwise and counterclockwise streams may share model parameters while maintaining independent hidden states. Parent histories are used only by the topology Mamba and do not enter the earlier independent contour scans, allowing all contours to be encoded in parallel.

### Architecture

```text
1. Structural intensity map and closed-contour forest
                         ↓
2. Normal scans from outside contours to sampled points
                         ↓
3. Independent bidirectional full-loop scans for all contours
                         ↓
4. One local state h_C for each contour
                         ↓
5. Recursive low-to-high propagation through containment branches
                         ↓
6. One branch state h_peak for each peak
                         ↓
7. Low-to-high peak-intensity scan
                         ↓
8. Final state h → image classification
```

### Future Extensions

- **Trainable structural map:** A convolutional module can generate a same-resolution structural intensity map, while softened level representations connect it to the sampling stage and allow the classification loss to reach the map generator.

- **Principal-direction ring scanning:** Concentric rings are built around a contour point, a local principal direction aligns the start, end, and aggregation position of each ring, bidirectional Mamba scans encode every ring, and the ring states are propagated from the outside toward the center.

- **Dense inter-contour scanning:** Horizontal and vertical scans can be introduced between neighboring contours to capture regions missed by sparse normal segments, after which the dense states are aggregated into nearby contour points.

- **Trainable sampling strategy:** The current curvature- and spacing-based sampling density can be replaced by a learned strategy that dynamically selects sampling locations for each task.

- **Parallel tree scan:** A tree-parallel topology Mamba can be explored to reduce sequential waiting across contour depths.
