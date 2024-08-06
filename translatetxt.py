# 打开原始文件
with open("coco_val_ITSDT.txt", "r",encoding="utf-8") as infile:
    # 读取每一行内容
    lines = infile.readlines()

# 打开新文件
with open("new_coco_val_ITSDT.txt", "w",encoding="utf-8") as outfile:
    # 遍历每一行内容
    for line in lines:
        # 替换原始路径为新路径
        new_line = line.replace("/home/public/ITSDT/images/", "E://datasets//tiny_detection//ITDST//images//",)
        # 写入新文件
        outfile.write(new_line)
