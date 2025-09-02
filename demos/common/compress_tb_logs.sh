#!/bin/bash

# ==============================================================================
# compress_tb_logs_with_stats.sh (v3 - Final Fix)
#
# Description:
#   Finds all TensorBoard event files (*.tfevents.*) in a source directory,
#   creates a .tar.xz archive with correct relative paths, and displays a
#   final summary. This version fixes the "Cannot stat" error.
#
# Usage:
#   ./compress_tb_logs_with_stats.sh <source_directory> <output_filename.tar.xz>
#
# ==============================================================================

# --- 1. 参数和基本检查 ---

if [ "$#" -ne 2 ]; then
    echo "❌ Error: Invalid number of arguments."
    echo "Usage: $0 <source_directory> <output_filename.tar.xz>"
    exit 1
fi

SOURCE_DIR="$1"
OUTPUT_FILE="$2"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "❌ Error: Source directory '$SOURCE_DIR' not found."
    exit 1
fi

# --- 2. 路径处理 ---
SOURCE_DIR_ABS=$(realpath "$SOURCE_DIR")
PARENT_DIR=$(dirname "$SOURCE_DIR_ABS")
TARGET_DIR_NAME=$(basename "$SOURCE_DIR_ABS")

# --- 3. 压缩前统计 ---

echo "📊 Calculating pre-compression statistics..."

FILE_COUNT=$( (cd "$PARENT_DIR" && find "$TARGET_DIR_NAME" -name "events.out.tfevents.*" -type f -print0 | tr -cd '\0' | wc -c) )

if [ "$FILE_COUNT" -eq 0 ]; then
    echo "⚠️ Warning: No matching files ('events.out.tfevents.*') found. Nothing to do."
    exit 0
fi

SIZE_BEFORE_BYTES=$( (cd "$PARENT_DIR" && find "$TARGET_DIR_NAME" -name "events.out.tfevents.*" -type f -print0 | xargs -0 -I {} stat --printf="%s\n" "{}" | awk '{s+=$1} END {print s}') )
SIZE_BEFORE_HUMAN=$(numfmt --to=iec-i --suffix=B --format="%.2f" "$SIZE_BEFORE_BYTES")

echo "---------------------------------"
echo "📋 Pre-compression Summary"
echo "   - Target directory:  $TARGET_DIR_NAME"
echo "   - Total files found: $FILE_COUNT"
echo "   - Total size:        $SIZE_BEFORE_HUMAN ($SIZE_BEFORE_BYTES bytes)"
echo "---------------------------------"
echo ""
echo "🚀 Starting compression..."
echo "ℹ️  The process will run silently. Please be patient, this may take a while..."

# --- 4. 执行压缩 (关键修正) ---

# **修正点**: 将 tar 命令移入子 Shell `( )` 中，并移除 tar 的 -C 参数。
# 这样 find 和 tar 都在 PARENT_DIR 目录中执行，路径完美匹配。
(cd "$PARENT_DIR" && find "$TARGET_DIR_NAME" -name "events.out.tfevents.*" -type f -print0 | tar --null -c --files-from -) | \
    xz -T0 -9e - > "$OUTPUT_FILE"

if [ ! -s "$OUTPUT_FILE" ]; then
    echo "❌ Error: Compression failed or the output file is empty."
    exit 1
fi

# --- 5. 压缩后统计与总结 ---

SIZE_AFTER_BYTES=$(stat -c %s "$OUTPUT_FILE")
SIZE_AFTER_HUMAN=$(numfmt --to=iec-i --suffix=B --format="%.2f" "$SIZE_AFTER_BYTES")
COMPRESSION_RATIO=$(awk "BEGIN {if ($SIZE_AFTER_BYTES > 0) {printf \"%.2f\", $SIZE_BEFORE_BYTES / $SIZE_AFTER_BYTES} else {print \"inf\"}}")

echo ""
echo "---------------------------------"
echo "✅ Final Compression Report"
echo "---------------------------------"
echo "   - Total files compressed: $FILE_COUNT"
echo "   - Size before compression:  $SIZE_BEFORE_HUMAN"
echo "   - Size after compression:   $SIZE_AFTER_HUMAN"
echo "   - Compression ratio:        ${COMPRESSION_RATIO}:1"
echo "   - Archive saved to:         $(realpath "$OUTPUT_FILE")"
echo "---------------------------------"