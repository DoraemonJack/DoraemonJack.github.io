// 文本高亮转换脚本
// 将简短的标签转换为完整的HTML标签

function convertHighlightTags(text) {
    // 定义颜色映射
    const colorMap = {
        'r': '#e74c3c',  // 红色
        'b': '#3498db',  // 蓝色
        'g': '#27ae60',  // 绿色
        'y': '#f39c12',  // 黄色
        'o': '#e67e22',  // 橙色
        'p': '#9b59b6',  // 紫色
        'k': '#2c3e50',  // 深灰色（关键词）
        'i': '#e74c3c'   // 红色（重要）
    };
    
    // 转换颜色标签 <r>text</r> → <span style="color: #e74c3c;">text</span>
    for (const [tag, color] of Object.entries(colorMap)) {
        const regex = new RegExp(`<${tag}>(.*?)</${tag}>`, 'g');
        text = text.replace(regex, `<span style="color: ${color};">$1</span>`);
    }
    
    // 转换数学公式标签 <m>formula</m> → <span style="color: #e74c3c;">$formula$</span>
    text = text.replace(/<m>(.*?)<\/m>/g, '<span style="color: #e74c3c;">$$$1$$</span>');
    
    return text;
}

// 页面加载完成后自动转换
document.addEventListener('DOMContentLoaded', function() {
    // 查找所有需要转换的元素
    const elements = document.querySelectorAll('.highlight-convert');
    
    elements.forEach(element => {
        element.innerHTML = convertHighlightTags(element.innerHTML);
    });
});

// 也可以手动调用
window.convertHighlight = convertHighlightTags;
