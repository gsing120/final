
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tabs if they exist
    const tabs = document.querySelectorAll('.tab');
    if (tabs.length > 0) {
        tabs.forEach(tab => {
            tab.addEventListener('click', function() {
                // Remove active class from all tabs and tab contents
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                
                // Add active class to clicked tab and corresponding content
                this.classList.add('active');
                const tabId = this.getAttribute('data-tab');
                document.getElementById(tabId).classList.add('active');
            });
        });
    }
    
    // Initialize memory sliders if they exist
    const memorySliders = document.querySelectorAll('.memory-slider');
    if (memorySliders.length > 0) {
        memorySliders.forEach(slider => {
            const valueDisplay = slider.nextElementSibling;
            valueDisplay.textContent = slider.value;
            
            slider.addEventListener('input', function() {
                valueDisplay.textContent = this.value;
            });
        });
    }
    
    // Initialize strategy generation form if it exists
    const strategyForm = document.getElementById('strategy-form');
    if (strategyForm) {
        strategyForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('strategy-result').style.display = 'none';
            
            const formData = new FormData(this);
            
            fetch('/generate_strategy', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                
                if (data.success === false) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                document.getElementById('strategy-result').style.display = 'block';
                
                // Update recommendation
                const recommendationEl = document.getElementById('recommendation');
                recommendationEl.textContent = data.prediction.recommendation.toUpperCase();
                recommendationEl.className = 'recommendation ' + data.prediction.recommendation.toLowerCase();
                
                // Update metrics
                document.getElementById('trend').textContent = data.prediction.trend.charAt(0).toUpperCase() + data.prediction.trend.slice(1);
                document.getElementById('momentum').textContent = data.prediction.momentum.charAt(0).toUpperCase() + data.prediction.momentum.slice(1);
                document.getElementById('volatility').textContent = data.prediction.volatility.charAt(0).toUpperCase() + data.prediction.volatility.slice(1);
                document.getElementById('support').textContent = '$' + data.prediction.support;
                document.getElementById('resistance').textContent = '$' + data.prediction.resistance;
                document.getElementById('total-return').textContent = data.performance.total_return;
                document.getElementById('sharpe-ratio').textContent = data.performance.sharpe_ratio;
                document.getElementById('max-drawdown').textContent = data.performance.max_drawdown;
                document.getElementById('win-rate').textContent = data.performance.win_rate;
                
                // Update plots
                const plotsContainer = document.getElementById('plots');
                plotsContainer.innerHTML = '';
                data.plot_urls.forEach(url => {
                    const img = document.createElement('img');
                    img.src = url;
                    img.alt = 'Strategy Plot';
                    plotsContainer.appendChild(img);
                });
                
                // Update trades table
                const tradesBody = document.getElementById('trades').getElementsByTagName('tbody')[0];
                tradesBody.innerHTML = '';
                data.trades.forEach(trade => {
                    const row = tradesBody.insertRow();
                    row.insertCell(0).textContent = trade.date;
                    row.insertCell(1).textContent = trade.type;
                    row.insertCell(2).textContent = trade.price;
                    row.insertCell(3).textContent = trade.shares;
                    row.insertCell(4).textContent = trade.pnl || '';
                });
                
                // Update description
                document.getElementById('description').textContent = data.description;
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                alert('Error generating strategy: ' + error);
            });
        });
    }
    
    // Initialize backtest form if it exists
    const backtestForm = document.getElementById('backtest-form');
    if (backtestForm) {
        backtestForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            document.getElementById('backtest-loading').style.display = 'block';
            document.getElementById('backtest-result').style.display = 'none';
            
            const formData = new FormData(this);
            
            fetch('/run_backtest', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('backtest-loading').style.display = 'none';
                
                if (data.success === false) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                document.getElementById('backtest-result').style.display = 'block';
                
                // Update metrics
                document.getElementById('initial-capital').textContent = '$' + data.initial_capital.toFixed(2);
                document.getElementById('final-capital').textContent = '$' + data.final_capital.toFixed(2);
                document.getElementById('backtest-return').textContent = data.total_return;
                document.getElementById('annual-return').textContent = data.annual_return;
                document.getElementById('backtest-sharpe').textContent = data.sharpe_ratio;
                document.getElementById('backtest-drawdown').textContent = data.max_drawdown;
                document.getElementById('backtest-winrate').textContent = data.win_rate;
                
                // Update plots
                const plotsContainer = document.getElementById('backtest-plots');
                plotsContainer.innerHTML = '';
                data.plot_urls.forEach(url => {
                    const img = document.createElement('img');
                    img.src = url;
                    img.alt = 'Backtest Plot';
                    plotsContainer.appendChild(img);
                });
                
                // Update trades table
                const tradesBody = document.getElementById('backtest-trades').getElementsByTagName('tbody')[0];
                tradesBody.innerHTML = '';
                data.trades.forEach(trade => {
                    const row = tradesBody.insertRow();
                    row.insertCell(0).textContent = trade.date;
                    row.insertCell(1).textContent = trade.type;
                    row.insertCell(2).textContent = trade.price;
                    row.insertCell(3).textContent = trade.shares;
                    row.insertCell(4).textContent = trade.value;
                    if (trade.pnl) {
                        row.insertCell(5).textContent = trade.pnl;
                    } else {
                        row.insertCell(5).textContent = '';
                    }
                });
            })
            .catch(error => {
                document.getElementById('backtest-loading').style.display = 'none';
                alert('Error running backtest: ' + error);
            });
        });
    }
    
    // Initialize scanner form if it exists
    const scannerForm = document.getElementById('scanner-form');
    if (scannerForm) {
        scannerForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            document.getElementById('scanner-loading').style.display = 'block';
            document.getElementById('scanner-result').style.display = 'none';
            
            const formData = new FormData(this);
            
            fetch('/scan_market', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('scanner-loading').style.display = 'none';
                
                if (data.success === false) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                document.getElementById('scanner-result').style.display = 'block';
                
                // Update results table
                const resultsBody = document.getElementById('scanner-results').getElementsByTagName('tbody')[0];
                resultsBody.innerHTML = '';
                data.results.forEach(result => {
                    const row = resultsBody.insertRow();
                    row.insertCell(0).textContent = result.ticker;
                    
                    const signalCell = row.insertCell(1);
                    signalCell.textContent = result.signal;
                    if (result.signal === 'BUY') {
                        signalCell.style.color = '#155724';
                        signalCell.style.backgroundColor = '#d4edda';
                    } else if (result.signal === 'SELL') {
                        signalCell.style.color = '#721c24';
                        signalCell.style.backgroundColor = '#f8d7da';
                    }
                    
                    row.insertCell(2).textContent = result.strength;
                    row.insertCell(3).textContent = result.pattern;
                    row.insertCell(4).textContent = result.volume;
                });
            })
            .catch(error => {
                document.getElementById('scanner-loading').style.display = 'none';
                alert('Error scanning market: ' + error);
            });
        });
    }
    
    // Initialize journal form if it exists
    const journalForm = document.getElementById('journal-form');
    if (journalForm) {
        journalForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            
            fetch('/add_journal_entry', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success === false) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                // Add new entry to table
                const journalBody = document.getElementById('journal-entries').getElementsByTagName('tbody')[0];
                const row = journalBody.insertRow(0);
                row.insertCell(0).textContent = data.entry.date;
                row.insertCell(1).textContent = data.entry.ticker;
                row.insertCell(2).textContent = data.entry.action;
                row.insertCell(3).textContent = '$' + data.entry.price.toFixed(2);
                row.insertCell(4).textContent = data.entry.shares;
                row.insertCell(5).textContent = data.entry.reason;
                row.insertCell(6).textContent = data.entry.result;
                
                // Reset form
                journalForm.reset();
                
                // Show success message
                alert('Journal entry added successfully');
            })
            .catch(error => {
                alert('Error adding journal entry: ' + error);
            });
        });
    }
    
    // Initialize memory settings form if it exists
    const memoryForm = document.getElementById('memory-form');
    if (memoryForm) {
        memoryForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            
            fetch('/update_memory_settings', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success === false) {
                    alert('Error: ' + data.message);
                    return;
                }
                
                // Show success message
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert alert-success';
                alertDiv.textContent = data.message;
                
                const formContainer = document.querySelector('.card-body');
                formContainer.insertBefore(alertDiv, formContainer.firstChild);
                
                // Remove alert after 3 seconds
                setTimeout(() => {
                    alertDiv.remove();
                }, 3000);
            })
            .catch(error => {
                alert('Error updating memory settings: ' + error);
            });
        });
    }
});
        