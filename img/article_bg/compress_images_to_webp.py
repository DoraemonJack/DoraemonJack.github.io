#!/usr/bin/env python3
"""
批量将图片压缩为WebP格式的脚本
支持PNG、JPG、JPEG、GIF、BMP等常见格式
"""

import os
import sys
from pathlib import Path
from PIL import Image
import argparse
from typing import List, Tuple
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 支持的图片格式
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}

class ImageCompressor:
    def __init__(self, quality: int = 75, recursive: bool = True, keep_original: bool = True):
        """
        初始化图片压缩器
        
        Args:
            quality: WebP压缩质量 (0-100)，默认75
            recursive: 是否递归处理子文件夹，默认True
            keep_original: 是否保留原始文件，默认True
        """
        self.quality = max(0, min(100, quality))  # 确保质量在0-100之间
        self.recursive = recursive
        self.keep_original = keep_original
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'original_size': 0,
            'compressed_size': 0
        }
    
    def get_image_files(self, folder_path: str) -> List[Path]:
        """获取文件夹中的所有图片文件"""
        folder = Path(folder_path)
        
        if not folder.exists():
            raise ValueError(f"文件夹不存在: {folder_path}")
        
        if not folder.is_dir():
            raise ValueError(f"路径不是文件夹: {folder_path}")
        
        # 获取图片文件列表
        if self.recursive:
            image_files = [
                f for f in folder.rglob('*') 
                if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
            ]
        else:
            image_files = [
                f for f in folder.glob('*') 
                if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
            ]
        
        return sorted(image_files)
    
    def compress_image(self, input_path: Path) -> Tuple[bool, str]:
        """
        压缩单个图片
        
        Args:
            input_path: 输入图片路径
            
        Returns:
            (成功, 消息) 元组
        """
        try:
            # 生成输出文件路径
            output_path = input_path.with_suffix('.webp')
            
            # 如果输出文件已存在且输入是webp格式，跳过
            if input_path.suffix.lower() == '.webp' and output_path.exists():
                self.stats['skipped'] += 1
                return True, f"跳过 (已是WebP格式): {input_path.name}"
            
            # 获取原始文件大小
            original_size = input_path.stat().st_size
            
            # 打开图片
            with Image.open(input_path) as img:
                # 转换RGBA到RGB (WebP处理问题)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # 创建白色背景
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 保存为WebP格式
                img.save(output_path, 'WEBP', quality=self.quality, method=6)
            
            # 获取压缩后的文件大小
            compressed_size = output_path.stat().st_size
            
            # 删除原始文件（如果不保留）
            if not self.keep_original and input_path != output_path:
                input_path.unlink()
            
            # 计算压缩率
            compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            
            # 更新统计信息
            self.stats['success'] += 1
            self.stats['original_size'] += original_size
            self.stats['compressed_size'] += compressed_size
            
            size_reduction = (original_size - compressed_size) / 1024  # KB
            
            message = (
                f"✓ {input_path.name} → {output_path.name} "
                f"({original_size/1024:.1f}KB → {compressed_size/1024:.1f}KB, "
                f"减少{compression_ratio:.1f}%)"
            )
            
            return True, message
            
        except Exception as e:
            self.stats['failed'] += 1
            return False, f"✗ {input_path.name} 失败: {str(e)}"
    
    def compress_folder(self, folder_path: str) -> None:
        """压缩整个文件夹中的所有图片"""
        try:
            image_files = self.get_image_files(folder_path)
            
            if not image_files:
                logger.warning(f"文件夹中未找到图片文件: {folder_path}")
                return
            
            self.stats['total'] = len(image_files)
            logger.info(f"找到 {len(image_files)} 个图片文件")
            logger.info(f"压缩质量: {self.quality}, 保留原始文件: {self.keep_original}")
            logger.info("-" * 80)
            
            for idx, image_path in enumerate(image_files, 1):
                success, message = self.compress_image(image_path)
                logger.info(f"[{idx}/{len(image_files)}] {message}")
            
            logger.info("-" * 80)
            self._print_summary()
            
        except Exception as e:
            logger.error(f"发生错误: {str(e)}")
            sys.exit(1)
    
    def _print_summary(self) -> None:
        """打印压缩统计摘要"""
        logger.info("=" * 80)
        logger.info("压缩完成! 统计摘要:")
        logger.info(f"  总文件数:     {self.stats['total']}")
        logger.info(f"  成功:         {self.stats['success']}")
        logger.info(f"  失败:         {self.stats['failed']}")
        logger.info(f"  跳过:         {self.stats['skipped']}")
        
        if self.stats['original_size'] > 0:
            total_ratio = (1 - self.stats['compressed_size'] / self.stats['original_size']) * 100
            original_mb = self.stats['original_size'] / (1024 * 1024)
            compressed_mb = self.stats['compressed_size'] / (1024 * 1024)
            saved_mb = original_mb - compressed_mb
            
            logger.info(f"  原始总大小:   {original_mb:.2f} MB")
            logger.info(f"  压缩后大小:   {compressed_mb:.2f} MB")
            logger.info(f"  节省空间:     {saved_mb:.2f} MB ({total_ratio:.1f}%)")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='批量将图片压缩为WebP格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 压缩当前文件夹中的所有图片，质量75
  python compress_images_to_webp.py .
  
  # 压缩指定文件夹，质量90，不保留原始文件
  python compress_images_to_webp.py ./images -q 90 --delete-original
  
  # 只压缩当前文件夹（不递归），质量60
  python compress_images_to_webp.py ./images --no-recursive -q 60
        """
    )
    
    parser.add_argument(
        'folder',
        help='要处理的文件夹路径'
    )
    
    parser.add_argument(
        '-q', '--quality',
        type=int,
        default=75,
        help='WebP压缩质量 (0-100)，默认75，数字越高质量越好但文件越大'
    )
    
    parser.add_argument(
        '-r', '--no-recursive',
        action='store_true',
        help='不递归处理子文件夹，只处理指定文件夹'
    )
    
    parser.add_argument(
        '-d', '--delete-original',
        action='store_true',
        help='删除原始文件，只保留WebP版本'
    )
    
    args = parser.parse_args()
    
    # 验证质量参数
    if not (0 <= args.quality <= 100):
        logger.error("质量参数必须在0-100之间")
        sys.exit(1)
    
    # 创建压缩器实例
    compressor = ImageCompressor(
        quality=args.quality,
        recursive=not args.no_recursive,
        keep_original=not args.delete_original
    )
    
    # 执行压缩
    compressor.compress_folder(args.folder)


if __name__ == '__main__':
    main()
