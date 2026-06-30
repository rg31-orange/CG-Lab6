# 实验六：基于 PyTorch3D 的三维网格模型渲染

## 1. 实验简介

本实验使用 **PyTorch3D** 对三维网格模型进行加载、处理与渲染。实验中读取 `cow.obj` 三维模型文件，提取模型的顶点和面片信息，并通过中心化、归一化等预处理操作，将模型调整到适合渲染的空间范围内。

在完成模型加载后，实验使用 PyTorch3D 构建 `Meshes` 对象，并设置相机、光源、光栅化器和着色器，最终实现对牛模型的多视角渲染、旋转动态渲染以及不同光照条件下的渲染对比。

---

## 2. 实验环境

本实验主要依赖以下环境：

- Python 3.12
- PyTorch
- PyTorch3D
- CUDA，可选
- Jupyter Notebook

由于 PyTorch3D 对 PyTorch、Python 和 CUDA 版本有一定要求，建议在课程提供的云端实验环境或已经配置好的 Python 环境中运行。

---

## 3. 文件结构

```text
lab6/
├── README.md
├── requirements.txt
├── lab6.ipynb
├── cow.obj
└── outputs/
    ├── cow_views_compare.png
    ├── cow_lights_compare.png
    ├── light_top.png
    └── cow_rotation.gif
```

其中：

| 文件 | 说明 |
|---|---|
| `README.md` | 实验六说明文档 |
| `requirements.txt` | 实验所需 Python 依赖 |
| `lab6.ipynb` | 实验六的 Jupyter Notebook 源代码 |
| `cow.obj` | 输入的三维牛模型文件 |
| `outputs/cow_views_compare.png` | 不同相机视角下的渲染结果 |
| `outputs/cow_lights_compare.png` | 不同光照位置下的渲染结果 |
| `outputs/light_top.png` | 顶部光照条件下的单张渲染结果 |
| `outputs/cow_rotation.gif` | 牛模型绕固定中心旋转的动态渲染结果 |

---

## 4. 实验原理

### 4.1 OBJ 模型加载

OBJ 是常见的三维模型文件格式，通常包含顶点、面片、法向量、纹理坐标和材质文件引用等信息。本实验使用 PyTorch3D 中的 `load_obj` 函数读取模型：

```python
verts, faces, aux = load_obj(obj_path, load_textures=False)
```

其中：

- `verts` 表示模型顶点坐标；
- `faces` 表示模型面片索引；
- `aux` 保存材质、纹理等附加信息。

由于本实验主要关注三维几何结构和基础渲染效果，因此设置 `load_textures=False`，避免缺少 `.mtl` 材质文件时产生不必要的警告。

---

### 4.2 模型中心化与归一化

为了让模型位于相机视野中心，并使其尺寸适合渲染，需要对顶点坐标进行中心化和归一化处理：

```python
verts = verts - verts.mean(0)
verts = verts / verts.abs().max()
```

中心化操作可以将模型移动到坐标原点附近，归一化操作可以将模型缩放到统一尺度。这样可以避免模型因为坐标偏移或尺寸过大而无法正常显示。

---

### 4.3 构建 Meshes 对象

PyTorch3D 使用 `Meshes` 类表示三维网格模型：

```python
cow_mesh = Meshes(verts=[verts], faces=[faces_idx])
```

该对象包含模型的顶点、面片以及可选的纹理和颜色信息，是后续渲染流程中的核心输入。

---

### 4.4 顶点颜色设置

由于实验中没有使用 `.mtl` 材质文件，因此手动为模型设置顶点颜色。通过 `TexturesVertex` 可以为每个顶点指定颜色：

```python
verts_rgb = torch.ones_like(verts)[None]
verts_rgb[:, :, 0] = 0.85
verts_rgb[:, :, 1] = 0.65
verts_rgb[:, :, 2] = 0.45

textures = TexturesVertex(verts_features=verts_rgb)
cow_mesh.textures = textures
```

这样即使没有材质文件，也可以正常渲染出带颜色的三维模型。

---

### 4.5 PyTorch3D 渲染流程

PyTorch3D 的基本渲染流程包括：

1. 设置相机；
2. 设置光源；
3. 设置光栅化参数；
4. 设置着色器；
5. 调用渲染器生成图像。

本实验使用透视相机 `FoVPerspectiveCameras`，并通过 `look_at_view_transform` 设置不同观察角度：

```python
R, T = look_at_view_transform(
    dist=2.7,
    elev=20.0,
    azim=30.0,
    device=device,
)
```

其中：

- `dist` 表示相机与物体之间的距离；
- `elev` 表示相机的俯仰角；
- `azim` 表示相机绕物体旋转的方位角。

---

## 5. 实验步骤

### 5.1 读取模型

首先读取 `cow.obj` 文件，并获得顶点和面片数据：

```python
verts, faces, aux = load_obj(obj_path, load_textures=False)

faces_idx = faces.verts_idx.to(device)
verts = verts.to(device)

verts = verts - verts.mean(0)
verts = verts / verts.abs().max()

cow_mesh = Meshes(verts=[verts], faces=[faces_idx])

print("Cow vertices:", verts.shape)
print("Cow faces:", faces_idx.shape)
```

实验运行后输出：

```text
Cow vertices: torch.Size([2930, 3])
Cow faces: torch.Size([5856, 3])
```

说明模型共包含 2930 个顶点和 5856 个三角面片。

---

### 5.2 添加顶点颜色

模型加载完成后，为牛模型设置浅棕色顶点颜色：

```python
verts_rgb = torch.ones_like(verts)[None]
verts_rgb[:, :, 0] = 0.85
verts_rgb[:, :, 1] = 0.65
verts_rgb[:, :, 2] = 0.45

textures = TexturesVertex(verts_features=verts_rgb)
cow_mesh.textures = textures
```

---

### 5.3 设置渲染参数

设置输出图像尺寸、光栅化参数以及背景颜色：

```python
image_size = 512

raster_settings = RasterizationSettings(
    image_size=image_size,
    blur_radius=0.0,
    faces_per_pixel=1,
)

blend_params = BlendParams(
    background_color=(1.0, 1.0, 1.0),
)
```

---

### 5.4 设置相机与光源

通过 `look_at_view_transform` 设置相机位置和方向：

```python
R, T = look_at_view_transform(
    dist=2.7,
    elev=20.0,
    azim=30.0,
    device=device,
)

cameras = FoVPerspectiveCameras(
    device=device,
    R=R,
    T=T,
)
```

设置点光源：

```python
lights = PointLights(
    device=device,
    location=[[0.0, 2.0, 2.0]],
)
```

---

### 5.5 构建渲染器

使用 `MeshRenderer`、`MeshRasterizer` 和 `SoftPhongShader` 构建渲染器：

```python
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings,
    ),
    shader=SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=lights,
        blend_params=blend_params,
    ),
)
```

---

### 5.6 多视角渲染

通过改变相机的 `azim` 和 `elev` 参数，分别从正面、侧面、顶部和斜向视角观察模型：

```python
view_settings = [
    {"name": "front", "dist": 2.7, "elev": 0.0, "azim": 0.0},
    {"name": "side", "dist": 2.7, "elev": 0.0, "azim": 90.0},
    {"name": "top", "dist": 2.7, "elev": 70.0, "azim": 0.0},
    {"name": "diagonal", "dist": 2.7, "elev": 25.0, "azim": 45.0},
]
```

---

### 5.7 旋转动态渲染

为了更直观地展示三维模型的空间结构，实验通过不断改变相机方位角 `azim`，生成多帧图像，并将其保存为 GIF 动图：

```python
num_frames = 36

for i in range(num_frames):
    azim = 360.0 * i / num_frames
```

最终生成：

```text
outputs/cow_rotation.gif
```

---

### 5.8 不同光照条件渲染

通过设置不同的点光源位置，观察光照方向对模型表面明暗和高光分布的影响：

```python
light_settings = [
    {"name": "light_front", "location": [[0.0, 0.0, 3.0]]},
    {"name": "light_top", "location": [[0.0, 3.0, 0.0]]},
    {"name": "light_left", "location": [[-3.0, 2.0, 2.0]]},
    {"name": "light_right", "location": [[3.0, 2.0, 2.0]]},
]
```

---

## 6. 实验结果

### 6.1 旋转动态渲染结果

下图展示了牛模型在相机环绕视角下的动态渲染效果。实验通过不断改变相机的方位角 `azim`，从多个角度对模型进行渲染，并将连续帧保存为 GIF 动图。

![牛模型旋转渲染结果](./outputs/cow_rotation.gif)

从动态结果可以看出，模型在不同方位角下能够保持稳定显示，说明模型的中心化、归一化以及相机参数设置较为合理。该结果直观展示了三维模型的整体结构和空间形态。

---

### 6.2 不同视角渲染结果

下图展示了牛模型在不同相机视角下的渲染结果，包括正面、侧面、顶部和斜视角。

![不同视角渲染结果](./outputs/cow_views_compare.png)

从结果可以看出，不同相机参数会显著影响模型在图像中的呈现方式。正面视角能够观察模型后部结构，侧面视角能够更清楚地观察身体轮廓，顶部视角则展示了模型的俯视形状，斜视角可以较全面地呈现模型的三维外形。

---

### 6.3 不同光照条件渲染结果

下图展示了在不同光源位置下的渲染效果，包括前方光源、顶部光源、左侧光源和右侧光源。

![不同光照条件渲染结果](./outputs/cow_lights_compare.png)

可以看到，当光源位置发生变化时，模型表面的高光区域和阴影分布也随之改变。光照方向会影响三维物体的明暗关系，从而增强模型的立体感。

---

### 6.4 顶部光照单张渲染结果

下图为顶部光照条件下的单张渲染结果：

![顶部光照渲染结果](./outputs/light_top.png)

在顶部光照条件下，模型上方区域更加明亮，身体下方区域相对较暗，体现了光源方向对渲染结果的影响。

---

## 7. 结果分析

通过本实验可以观察到以下现象：

1. **相机角度决定观察方向**  
   改变 `azim` 和 `elev` 参数，可以从不同方向观察同一个三维模型。

2. **光源位置影响明暗分布**  
   当光源位于模型前方、上方、左侧或右侧时，模型表面的高光和阴影区域会发生明显变化。

3. **模型预处理对渲染效果很重要**  
   中心化和归一化操作可以保证模型位于视野中心，并避免模型过大或过小导致显示异常。

4. **材质文件不是必须的**  
   即使缺少 `.mtl` 文件，也可以通过 `TexturesVertex` 手动设置模型颜色，从而完成基础渲染。

5. **动态渲染能够更完整展示三维结构**  
   相比单张静态图片，旋转 GIF 可以从连续视角展示模型外形，使模型的整体空间结构更加直观。

---

## 8. 运行方式

### 8.1 安装依赖

首先安装基础依赖：

```bash
pip install -r requirements.txt
```

如果 PyTorch3D 无法直接通过 pip 安装，可以使用课程实验环境中提供的安装方式，或从源码安装：

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

如果 GitHub 访问较慢，也可以使用课程指定的镜像源，例如：

```bash
pip install "git+https://gitee.com/hongwenzhang/pytorch3d.git"
```

---

### 8.2 运行 Notebook

在 Jupyter Notebook 中打开：

```text
lab6.ipynb
```

然后依次运行所有代码单元即可。

运行完成后，渲染结果会保存在：

```text
outputs/
```

---

## 9. 可能遇到的问题

### 9.1 提示缺少 cow.mtl

如果运行时出现：

```text
UserWarning: Mtl file does not exist: cow.mtl
```

这是因为 `cow.obj` 文件中引用了材质文件 `cow.mtl`，但当前目录下没有该文件。该提示只是警告，不影响模型顶点和面片的读取。

可以使用以下方式避免该警告：

```python
verts, faces, aux = load_obj(obj_path, load_textures=False)
```

---

### 9.2 PyTorch3D 安装失败

PyTorch3D 对 Python、PyTorch 和 CUDA 版本较敏感。如果安装失败，建议：

1. 使用课程提供的云端实验环境；
2. 确认 PyTorch 版本与 CUDA 版本匹配；
3. 使用源码安装 PyTorch3D；
4. 如果不使用 GPU，可在 CPU 环境下运行基础渲染代码，但速度可能较慢。

---

### 9.3 GIF 文件过大

如果 `cow_rotation.gif` 文件过大，可以减少帧数或降低渲染分辨率，例如：

```python
num_frames = 18
image_size = 256
```

这样可以明显减小 GIF 文件大小，方便上传到 GitHub。

---

## 10. 实验总结

本实验完成了基于 PyTorch3D 的三维模型加载、预处理和渲染。通过实验掌握了 OBJ 模型读取、网格对象构建、顶点颜色设置、相机设置、光照设置、多视角渲染以及动态 GIF 生成的基本流程。

实验结果表明，PyTorch3D 可以方便地实现三维模型渲染和图像生成。相机参数和光源位置是影响最终渲染效果的重要因素。通过调整这些参数，可以从不同方向和不同光照条件下观察三维模型。

本实验为后续学习三维视觉、三维重建、可微渲染和神经渲染等内容奠定了基础。
