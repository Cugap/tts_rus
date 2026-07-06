import os
import sys
from pathlib import Path

def main():
    print("Начинаем подготовку приложения к запуску...")
    
    # 1. Установка базовых зависимостей, если нужно
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Устанавливаем huggingface_hub для скачивания модели...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import snapshot_download

    # 2. Скачивание модели
    model_id = "utrobinmv/tts_ru_free_hf_vits_high_multispeaker"
    local_dir = Path(__file__).resolve().parent / "tts_ru_free_hf_vits_high_multispeaker"
    
    print(f"\nСкачивание модели {model_id}...")
    print(f"Папка назначения: {local_dir}")
    
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # Скачиваем реальные файлы, а не симлинки (важно для Windows)
        ignore_patterns=["*.msgpack", "*.h5", "coreml/*", "onnx/*"] # Игнорируем ненужные форматы
    )
    print("Модель успешно скачана!")
    
    # 3. Создание нужных директорий (на всякий случай)
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "uploads").mkdir(exist_ok=True)

    print("\nПодготовка завершена! Теперь вы можете запустить сервер командой:")
    print("python -m uvicorn app.main:app --host 127.0.0.1 --port 8000")

if __name__ == "__main__":
    main()
