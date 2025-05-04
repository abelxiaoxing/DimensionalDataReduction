import sys
import os
import importlib

# 确保 utils 目录在 Python 路径中
# 这对于 Nuitka 打包后正确找到模块很重要
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'utils'))

# 定义可用的演示及其对应的脚本
DEMOS = {
    'mnist': 'mnist_inference',
    'raman': 'raman_reduction',
    'wine': 'wine_inference'
}

def run_demo(demo_name):
    """导入并运行指定演示的主函数"""
    if demo_name not in DEMOS:
        print(f"错误：无效的演示名称 '{demo_name}'")
        print("可用演示：", ', '.join(DEMOS.keys()))
        return

    module_name = DEMOS[demo_name]
    try:
        print(f"正在加载并运行 {demo_name} 演示 ({module_name}.py)...")
        # 动态导入模块
        module = importlib.import_module(module_name)
        # 检查模块是否有 main 函数
        if hasattr(module, 'main') and callable(module.main):
            module.main() # 调用 main 函数
        else:
            print(f"错误：在 {module_name}.py 中未找到可执行的 'main' 函数。")
    except ImportError as e:
        print(f"错误：无法导入模块 '{module_name}'. 错误信息：{e}")
        print("请确保该文件存在并且所有依赖项都已安装。")
    except Exception as e:
        print(f"运行 {demo_name} 演示时出错：{e}")

def main():
    """主函数，处理用户输入并启动选择的演示"""
    print("欢迎使用多维数据降维演示启动器！")
    print("可用演示：")
    for key in DEMOS:
        print(f"- {key}")

    while True:
        choice = input("请输入您想运行的演示名称 (或输入 'exit' 退出): ").strip().lower()

        if choice == 'exit':
            print("正在退出启动器。")
            break
        elif choice in DEMOS:
            run_demo(choice)
        else:
            print("无效的选择，请重试。")

if __name__ == "__main__":
    main()