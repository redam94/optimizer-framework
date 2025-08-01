// Custom JavaScript for documentation

document.addEventListener('DOMContentLoaded', function() {
    // Add copy buttons to code blocks
    const codeBlocks = document.querySelectorAll('div.highlight pre');
    
    codeBlocks.forEach(function(codeBlock) {
        // Create copy button
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-button';
        copyButton.textContent = 'Copy';
        
        // Add click handler
        copyButton.addEventListener('click', function() {
            const code = codeBlock.textContent;
            navigator.clipboard.writeText(code).then(function() {
                copyButton.textContent = 'Copied!';
                setTimeout(function() {
                    copyButton.textContent = 'Copy';
                }, 2000);
            });
        });
        
        // Add button to code block
        const highlightDiv = codeBlock.parentNode;
        highlightDiv.style.position = 'relative';
        highlightDiv.appendChild(copyButton);
    });
    
    // Add anchor links to headers
    const headers = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
    
    headers.forEach(function(header) {
        if (header.id) {
            const anchor = document.createElement('a');
            anchor.className = 'headerlink';
            anchor.href = '#' + header.id;
            anchor.innerHTML = '¶';
            anchor.title = 'Permalink to this headline';
            header.appendChild(anchor);
        }
    });
    
    // Smooth scrolling for internal links
    document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});