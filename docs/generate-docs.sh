#!/bin/bash

# Script to generate HTML documentation from markdown files with Mermaid diagram rendering
# This script processes all .md files in the docs directory and creates HTML versions
# with properly rendered Mermaid diagrams using client-side rendering

set -e

DOCS_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$DOCS_DIR/html"

echo "================================================"
echo "Documentation Generator with Mermaid Support"
echo "================================================"
echo ""
echo "Working directory: $DOCS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# HTML template header with Mermaid.js support
create_html_header() {
    local title="$1"
    cat << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TITLE_PLACEHOLDER</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>
        // Initialize Mermaid
        mermaid.initialize({ 
            startOnLoad: true,
            theme: 'default',
            themeVariables: {
                primaryColor: '#fff',
                primaryTextColor: '#333',
                primaryBorderColor: '#7C8B9C',
                lineColor: '#5D6D7E',
                secondaryColor: '#006FBB',
                tertiaryColor: '#fff'
            },
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            }
        });
        
        // Convert code blocks with mermaid language to mermaid divs
        document.addEventListener('DOMContentLoaded', function() {
            // Handle both <pre><code class="language-mermaid"> and <pre class="mermaid"><code>
            const codeBlocks1 = document.querySelectorAll('pre code.language-mermaid');
            const codeBlocks2 = document.querySelectorAll('pre.mermaid code');
            const allCodeBlocks = [...codeBlocks1, ...codeBlocks2];
            
            allCodeBlocks.forEach(function(codeBlock) {
                const pre = codeBlock.parentElement;
                const mermaidDiv = document.createElement('div');
                mermaidDiv.className = 'mermaid';
                mermaidDiv.textContent = codeBlock.textContent;
                pre.parentNode.replaceChild(mermaidDiv, pre);
            });
            
            // Re-initialize mermaid after DOM changes
            mermaid.init();
        });
    </script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #f5f7fa;
        }
        .container {
            background: white;
            border-radius: 10px;
            padding: 40px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        h2 {
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
            font-size: 1.8em;
        }
        h3 {
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 1.4em;
        }
        h4 {
            color: #7f8c8d;
            margin-top: 20px;
            margin-bottom: 10px;
            font-size: 1.2em;
        }
        pre {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 14px;
            line-height: 1.5;
            margin: 20px 0;
        }
        code {
            background: #ecf0f1;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', monospace;
            font-size: 0.9em;
        }
        pre code {
            background: none;
            padding: 0;
            font-size: inherit;
            color: #ecf0f1;
        }
        .mermaid {
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            margin: 30px auto;
            text-align: center;
            overflow-x: auto;
            max-width: 100%;
        }
        blockquote {
            border-left: 4px solid #3498db;
            padding-left: 20px;
            margin: 20px 0;
            color: #555;
            font-style: italic;
            background: #f8f9fa;
            padding: 15px 20px;
            border-radius: 0 8px 8px 0;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background: #3498db;
            color: white;
        }
        tr:nth-child(even) {
            background: #f9f9f9;
        }
        ul, ol {
            margin: 15px 0;
            padding-left: 30px;
        }
        li {
            margin: 8px 0;
        }
        a {
            color: #3498db;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        .nav {
            background: #2c3e50;
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        .nav a {
            color: white;
            margin-right: 20px;
        }
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }
        .warning {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }
        .info {
            background: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }
        .success {
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }
        .error {
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }
    </style>
</head>
<body>
    <div class="container">
EOF
    echo "$1" | sed "s|TITLE_PLACEHOLDER|$title|g"
}

# HTML template footer
create_html_footer() {
    cat << 'EOF'
        <div class="footer">
            Generated with Mermaid diagram support
        </div>
    </div>
</body>
</html>
EOF
}

# Process a single markdown file
process_markdown_file() {
    local input_file="$1"
    local filename=$(basename "$input_file" .md)
    local output_file="$OUTPUT_DIR/${filename}.html"
    local title=$(echo "$filename" | sed 's/-/ /g' | sed 's/\b\(.\)/\u\1/g')
    
    echo "Processing: $input_file"
    echo "  Output: $output_file"
    
    # Create temporary file for processed markdown
    local temp_md="/tmp/${filename}_processed.md"
    
    # Clean up problematic HTML tags in mermaid blocks
    sed 's/<br\/>/\n/g' "$input_file" > "$temp_md"
    
    # Generate HTML body from markdown
    local html_body=$(pandoc "$temp_md" -t html --no-highlight 2>/dev/null || pandoc "$input_file" -t html --no-highlight)
    
    # Create full HTML document
    {
        create_html_header "$title"
        echo "$html_body"
        create_html_footer
    } > "$output_file"
    
    # Clean up temp file
    rm -f "$temp_md"
    
    echo "  ✓ Generated successfully"
}

# Create index page
create_index_page() {
    local index_file="$OUTPUT_DIR/index.html"
    
    echo "Creating index page..."
    
    {
        create_html_header "Documentation Index"
        cat << 'EOF'
        <h1>Documentation Index</h1>
        <p>Available documentation files:</p>
        <ul>
EOF
        
        # List all generated HTML files
        for html_file in "$OUTPUT_DIR"/*.html; do
            if [ -f "$html_file" ] && [ "$(basename "$html_file")" != "index.html" ]; then
                local filename=$(basename "$html_file" .html)
                local title=$(echo "$filename" | sed 's/-/ /g' | sed 's/\b\(.\)/\u\1/g')
                echo "            <li><a href=\"$(basename "$html_file")\">$title</a></li>"
            fi
        done
        
        echo "        </ul>"
        create_html_footer
    } > "$index_file"
    
    echo "  ✓ Index page created"
}

# Main processing
echo "Starting documentation generation..."
echo ""

# Find and process all markdown files
file_count=0
for md_file in "$DOCS_DIR"/*.md "$DOCS_DIR"/**/*.md; do
    if [ -f "$md_file" ]; then
        process_markdown_file "$md_file"
        ((file_count++))
    fi
done

# Create index page if files were processed
if [ $file_count -gt 0 ]; then
    create_index_page
    echo ""
    echo "================================================"
    echo "✓ Documentation generation complete!"
    echo "  Processed $file_count markdown files"
    echo "  Output directory: $OUTPUT_DIR"
    echo ""
    echo "To view the documentation:"
    echo "  open $OUTPUT_DIR/index.html"
    echo "================================================"
else
    echo "No markdown files found in $DOCS_DIR"
    exit 1
fi