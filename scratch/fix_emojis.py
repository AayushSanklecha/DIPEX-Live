import sys

file_path = r'c:\Users\sankl\Desktop\dipex_project\frontend\src\pages\RunPipeline.jsx'

replacements = {
    "âœ‰ï¸ ": "✉️",
    "ðŸŒ ": "🌐",
    "ðŸ ¥": "🏥",
    "ðŸ ¦": "🏦",
    "ðŸ›¡ï¸ ": "🛡️",
    "âš ï¸ ": "⚠️",
    "â ­ï¸ ": "➖",
    "ðŸ” ": "🔍",
    "ðŸ—³ï¸ ": "🗳️",
    "âš–ï¸ ": "⚖️",
    "ðŸ ·ï¸ ": "🏷️", # For categorical maybe?
    "ðŸ“ ": "📈"
}

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    for old, new in replacements.items():
        content = content.replace(old, new)
            
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("Fixed more emojis successfully!")
except Exception as e:
    print(f"Error: {e}")
