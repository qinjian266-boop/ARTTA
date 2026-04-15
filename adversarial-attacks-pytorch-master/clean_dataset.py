import os
import shutil

def clean_directory(root_dir, num_to_keep=10):
    """
    清理指定目录，确保每个子目录下只保留特定数量的文件。

    :param root_dir: 包含类别子目录的根文件夹路径。
    :param num_to_keep: 每个子目录中要保留的文件数量。
    """
    if not os.path.isdir(root_dir):
        print(f"错误：目录 '{root_dir}' 不存在。")
        return

    print(f"正在清理目录: {os.path.abspath(root_dir)}")
    print(f"将确保每个子目录中只保留 {num_to_keep} 个样本...")

    total_deleted_count = 0
    # 遍历根目录下的所有条目
    for class_name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)

        # 确保它是一个目录
        if os.path.isdir(class_dir):
            files = sorted([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])
            
            if len(files) > num_to_keep:
                files_to_delete = files[num_to_keep:]
                deleted_in_class = 0
                for file_to_delete in files_to_delete:
                    file_path = os.path.join(class_dir, file_to_delete)
                    try:
                        os.remove(file_path)
                        deleted_in_class += 1
                    except OSError as e:
                        print(f"删除文件 '{file_path}' 时出错: {e}")
                
                if deleted_in_class > 0:
                    print(f"在类别 '{class_name}' 中: 保留了 {num_to_keep} 个文件, 删除了 {deleted_in_class} 个。")
                    total_deleted_count += deleted_in_class

    if total_deleted_count > 0:
        print(f"\n清理完成。总共删除了 {total_deleted_count} 个多余的文件。")
    else:
        print("\n检查完成。所有子目录的文件数量均未超过限制，无需操作。")


if __name__ == '__main__':
    # --- 配置区域 ---
    # 请将此路径设置为您想要清理的文件夹
    # 默认指向之前生成对抗样本的输出目录
    target_directory = './imagenet-C-blur1-200classes-PGD-adv'

    # 设置每个类别（子目录）需要保留的样本数量
    samples_to_keep = 10
    # --- 配置结束 ---

    # 运行清理函数
    clean_directory(target_directory, samples_to_keep) 