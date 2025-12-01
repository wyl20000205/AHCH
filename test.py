from PIL import Image, ImageDraw
import os
from typing import Union, Tuple


def add_white_grid(
    image_path: str,
    num: Union[int, Tuple[int, int]],
    line_width: int = 1,
    save_path: str = None,
) -> str:
    """
    在图片上叠加白色网格（使用 Pillow）。

    参数：
        image_path: 输入图片路径
        num: 网格数量，int 表示 NxN；或 (rows, cols)
        line_width: 网格线宽（像素）
        save_path: 输出保存路径，默认与输入同目录，文件名后缀为 _grid

    返回：
        实际保存的文件路径
    """
    if isinstance(num, int):
        if num < 1:
            raise ValueError("num 必须 >= 1")
        rows = cols = num
    elif isinstance(num, (tuple, list)) and len(num) == 2:
        rows, cols = int(num[0]), int(num[1])
        if rows < 1 or cols < 1:
            raise ValueError("rows 和 cols 必须 >= 1")
    else:
        raise TypeError("num 必须是 int 或 (rows, cols)")

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    w, h = img.size
    row_step = h / rows
    col_step = w / cols

    # 画竖线
    for c in range(1, cols):
        x = int(round(c * col_step))
        draw.line([(x, 0), (x, h)], fill=(255, 255, 255), width=line_width)

    # 画横线
    for r in range(1, rows):
        y = int(round(r * row_step))
        draw.line([(0, y), (w, y)], fill=(255, 255, 255), width=line_width)

    if save_path is None:
        root, ext = os.path.splitext(image_path)
        save_path = f"{root}_grid{ext}"

    img.save(save_path)
    return save_path


if __name__ == "__main__":
    # 示例 1：生成 10x10 网格
    out1 = add_white_grid("ahph.jpg", num=4, line_width=4)
    print("保存到：", out1)

    # 示例 2：生成 8 行 x 12 列网格
    # out2 = add_white_grid(
    #     "fufu.jpg", num=(8, 12), line_width=1, save_path="output_8x12.jpg"
    # )
