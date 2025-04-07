import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Button, Form, Table, Badge, Spinner, Alert, Tabs, Tab } from 'react-bootstrap';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const StrategyBuilder = ({ onSave, existingStrategy = null }) => {
  const [strategy, setStrategy] = useState({
    name: '',
    description: '',
    type: 'trend_following',
    timeframe: '1d',
    entryConditions: [],
    exitConditions: [],
    riskManagement: {
      positionSize: 2, // percentage of portfolio
      stopLoss: 2, // percentage
      takeProfit: 4, // percentage
      maxOpenPositions: 5,
      trailingStop: false,
      trailingStopDistance: 1.5, // percentage
    },
    indicators: [],
    filters: {
      minPrice: 5,
      maxPrice: 1000,
      minVolume: 500000,
      sectors: [],
      excludedSymbols: []
    },
    backtestResults: null,
    isActive: false,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  });

  const [availableIndicators, setAvailableIndicators] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [backtestInProgress, setBacktestInProgress] = useState(false);
  const [backtestResults, setBacktestResults] = useState(null);
  const [activeTab, setActiveTab] = useState('general');
  const [conditionType, setConditionType] = useState('entry');
  const [newCondition, setNewCondition] = useState({
    indicator: '',
    comparator: '>',
    value: '',
    lookback: 1,
    secondaryIndicator: '',
    secondaryValue: '',
    useSecondaryIndicator: false,
  });
  const [newFilter, setNewFilter] = useState('');
  const [previewData, setPreviewData] = useState(null);

  // Load existing strategy if provided
  useEffect(() => {
    if (existingStrategy) {
      setStrategy(existingStrategy);
    }
    
    // Fetch available indicators
    fetchAvailableIndicators();
    
    // Generate preview data for visualization
    generatePreviewData();
  }, [existingStrategy]);

  const fetchAvailableIndicators = async () => {
    // In a real implementation, this would fetch from the backend
    // Simulating API call
    setLoading(true);
    
    setTimeout(() => {
      const indicators = [
        { id: 'sma', name: 'Simple Moving Average (SMA)', category: 'trend', parameters: ['period'] },
        { id: 'ema', name: 'Exponential Moving Average (EMA)', category: 'trend', parameters: ['period'] },
        { id: 'macd', name: 'MACD', category: 'momentum', parameters: ['fastPeriod', 'slowPeriod', 'signalPeriod'] },
        { id: 'rsi', name: 'Relative Strength Index (RSI)', category: 'momentum', parameters: ['period'] },
        { id: 'bbands', name: 'Bollinger Bands', category: 'volatility', parameters: ['period', 'stdDev'] },
        { id: 'atr', name: 'Average True Range (ATR)', category: 'volatility', parameters: ['period'] },
        { id: 'obv', name: 'On-Balance Volume (OBV)', category: 'volume', parameters: [] },
        { id: 'adx', name: 'Average Directional Index (ADX)', category: 'trend', parameters: ['period'] },
        { id: 'stoch', name: 'Stochastic Oscillator', category: 'momentum', parameters: ['kPeriod', 'dPeriod', 'slowing'] },
        { id: 'ichimoku', name: 'Ichimoku Cloud', category: 'trend', parameters: ['conversionPeriod', 'basePeriod', 'spanPeriod', 'displacement'] },
        { id: 'vwap', name: 'Volume Weighted Average Price (VWAP)', category: 'volume', parameters: ['period'] },
        { id: 'supertrend', name: 'SuperTrend', category: 'trend', parameters: ['period', 'multiplier'] },
        { id: 'keltner', name: 'Keltner Channels', category: 'volatility', parameters: ['period', 'atrPeriod', 'multiplier'] },
        { id: 'zigzag', name: 'ZigZag', category: 'pattern', parameters: ['percentage'] },
        { id: 'hma', name: 'Hull Moving Average (HMA)', category: 'trend', parameters: ['period'] },
        { id: 'vwma', name: 'Volume Weighted Moving Average (VWMA)', category: 'volume', parameters: ['period'] },
      ];
      
      setAvailableIndicators(indicators);
      setLoading(false);
    }, 500);
  };

  const generatePreviewData = () => {
    // Generate sample data for strategy visualization
    const dates = [];
    const prices = [];
    const indicators = {};
    
    // Generate 100 days of sample data
    const today = new Date();
    let price = 100;
    
    for (let i = 0; i < 100; i++) {
      const date = new Date(today);
      date.setDate(date.getDate() - (100 - i));
      dates.push(date.toISOString().split('T')[0]);
      
      // Random walk for price
      price = price + (Math.random() - 0.48) * 2;
      prices.push(price);
      
      // Generate indicator data
      if (i >= 20) {
        // SMA 20
        const sma20 = prices.slice(Math.max(0, i - 20), i).reduce((sum, p) => sum + p, 0) / Math.min(20, i);
        if (!indicators.sma20) indicators.sma20 = [];
        indicators.sma20.push(sma20);
        
        // SMA 50
        const sma50 = prices.slice(Math.max(0, i - 50), i).reduce((sum, p) => sum + p, 0) / Math.min(50, i);
        if (!indicators.sma50) indicators.sma50 = [];
        indicators.sma50.push(sma50);
        
        // RSI (simplified)
        const gains = [];
        const losses = [];
        for (let j = Math.max(0, i - 14); j < i; j++) {
          const change = prices[j] - prices[j - 1];
          if (change >= 0) {
            gains.push(change);
            losses.push(0);
          } else {
            gains.push(0);
            losses.push(Math.abs(change));
          }
        }
        
        const avgGain = gains.reduce((sum, g) => sum + g, 0) / gains.length;
        const avgLoss = losses.reduce((sum, l) => sum + l, 0) / losses.length;
        
        const rs = avgGain / (avgLoss === 0 ? 0.001 : avgLoss);
        const rsi = 100 - (100 / (1 + rs));
        
        if (!indicators.rsi) indicators.rsi = [];
        indicators.rsi.push(rsi);
      } else {
        if (!indicators.sma20) indicators.sma20 = [];
        if (!indicators.sma50) indicators.sma50 = [];
        if (!indicators.rsi) indicators.rsi = [];
        
        indicators.sma20.push(null);
        indicators.sma50.push(null);
        indicators.rsi.push(null);
      }
    }
    
    setPreviewData({
      dates,
      prices,
      indicators
    });
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    
    // Handle nested properties
    if (name.includes('.')) {
      const [parent, child] = name.split('.');
      setStrategy({
        ...strategy,
        [parent]: {
          ...strategy[parent],
          [child]: value
        }
      });
    } else {
      setStrategy({
        ...strategy,
        [name]: value
      });
    }
  };

  const handleNumberInputChange = (e) => {
    const { name, value } = e.target;
    const numValue = parseFloat(value);
    
    // Handle nested properties
    if (name.includes('.')) {
      const [parent, child] = name.split('.');
      setStrategy({
        ...strategy,
        [parent]: {
          ...strategy[parent],
          [child]: numValue
        }
      });
    } else {
      setStrategy({
        ...strategy,
        [name]: numValue
      });
    }
  };

  const handleCheckboxChange = (e) => {
    const { name, checked } = e.target;
    
    // Handle nested properties
    if (name.includes('.')) {
      const [parent, child] = name.split('.');
      setStrategy({
        ...strategy,
        [parent]: {
          ...strategy[parent],
          [child]: checked
        }
      });
    } else {
      setStrategy({
        ...strategy,
        [name]: checked
      });
    }
  };

  const handleNewConditionChange = (e) => {
    const { name, value } = e.target;
    setNewCondition({
      ...newCondition,
      [name]: value
    });
  };

  const handleAddCondition = () => {
    if (!newCondition.indicator) {
      setError('Please select an indicator for the condition');
      return;
    }
    
    const condition = {
      ...newCondition,
      id: `condition-${Date.now()}`
    };
    
    if (conditionType === 'entry') {
      setStrategy({
        ...strategy,
        entryConditions: [...strategy.entryConditions, condition]
      });
    } else {
      setStrategy({
        ...strategy,
        exitConditions: [...strategy.exitConditions, condition]
      });
    }
    
    // Reset new condition form
    setNewCondition({
      indicator: '',
      comparator: '>',
      value: '',
      lookback: 1,
      secondaryIndicator: '',
      secondaryValue: '',
      useSecondaryIndicator: false,
    });
    
    setSuccess(`Added new ${conditionType} condition`);
    setTimeout(() => setSuccess(null), 3000);
  };

  const handleRemoveCondition = (id, type) => {
    if (type === 'entry') {
      setStrategy({
        ...strategy,
        entryConditions: strategy.entryConditions.filter(c => c.id !== id)
      });
    } else {
      setStrategy({
        ...strategy,
        exitConditions: strategy.exitConditions.filter(c => c.id !== id)
      });
    }
  };

  const handleAddIndicator = (indicator) => {
    // Check if indicator already exists
    const exists = strategy.indicators.some(ind => ind.id === indicator.id);
    
    if (!exists) {
      // Create default parameters based on indicator definition
      const params = {};
      indicator.parameters.forEach(param => {
        // Set default values based on parameter name
        switch(param) {
          case 'period':
            params[param] = 14;
            break;
          case 'fastPeriod':
            params[param] = 12;
            break;
          case 'slowPeriod':
            params[param] = 26;
            break;
          case 'signalPeriod':
            params[param] = 9;
            break;
          case 'stdDev':
            params[param] = 2;
            break;
          case 'multiplier':
            params[param] = 3;
            break;
          case 'percentage':
            params[param] = 5;
            break;
          default:
            params[param] = 14;
        }
      });
      
      const newIndicator = {
        id: indicator.id,
        name: indicator.name,
        parameters: params
      };
      
      setStrategy({
        ...strategy,
        indicators: [...strategy.indicators, newIndicator]
      });
      
      setSuccess(`Added ${indicator.name} indicator`);
      setTimeout(() => setSuccess(null), 3000);
    } else {
      setError(`${indicator.name} is already added to the strategy`);
      setTimeout(() => setError(null), 3000);
    }
  };

  const handleRemoveIndicator = (id) => {
    setStrategy({
      ...strategy,
      indicators: strategy.indicators.filter(ind => ind.id !== id)
    });
  };

  const handleUpdateIndicatorParam = (indicatorId, param, value) => {
    const updatedIndicators = strategy.indicators.map(ind => {
      if (ind.id === indicatorId) {
        return {
          ...ind,
          parameters: {
            ...ind.parameters,
            [param]: parseFloat(value)
          }
        };
      }
      return ind;
    });
    
    setStrategy({
      ...strategy,
      indicators: updatedIndicators
    });
  };

  const handleAddExcludedSymbol = () => {
    if (newFilter && !strategy.filters.excludedSymbols.includes(newFilter)) {
      setStrategy({
        ...strategy,
        filters: {
          ...strategy.filters,
          excludedSymbols: [...strategy.filters.excludedSymbols, newFilter]
        }
      });
      setNewFilter('');
    }
  };

  const handleRemoveExcludedSymbol = (symbol) => {
    setStrategy({
      ...strategy,
      filters: {
        ...strategy.filters,
        excludedSymbols: strategy.filters.excludedSymbols.filter(s => s !== symbol)
      }
    });
  };

  const handleSectorChange = (sector, checked) => {
    if (checked) {
      setStrategy({
        ...strategy,
        filters: {
          ...strategy.filters,
          sectors: [...strategy.filters.sectors, sector]
        }
      });
    } else {
      setStrategy({
        ...strategy,
        filters: {
          ...strategy.filters,
          sectors: strategy.filters.sectors.filter(s => s !== sector)
        }
      });
    }
  };

  const handleRunBacktest = () => {
    setBacktestInProgress(true);
    setError(null);
    
    // Simulate backtest API call
    setTimeout(() => {
      // Generate mock backtest results
      const results = {
        summary: {
          totalTrades: 42,
          winningTrades: 25,
          losingTrades: 17,
          winRate: 59.52,
          profitFactor: 1.87,
          sharpeRatio: 1.42,
          maxDrawdown: 12.34,
          annualizedReturn: 18.76,
          totalReturn: 34.21
        },
        trades: Array(42).fill().map((_, i) => ({
          id: i + 1,
          symbol: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'][Math.floor(Math.random() * 5)],
          entryDate: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
          entryPrice: 100 + Math.random() * 200,
          exitDate: new Date(Date.now() - Math.random() * 15 * 24 * 60 * 60 * 1000).toISOString(),
          exitPrice: 100 + Math.random() * 200,
          quantity: Math.floor(Math.random() * 100) + 1,
          pnl: (Math.random() * 2 - 0.5) * 1000,
          pnlPercent: (Math.random() * 2 - 0.5) * 10,
          holdingPeriod: Math.floor(Math.random() * 30) + 1
        })),
        equityCurve: Array(100).fill().reduce((acc, _, i) => {
          const prevValue = acc.length > 0 ? acc[acc.length - 1].value : 10000;
          const change = (Math.random() - 0.45) * 200;
          return [...acc, {
            date: new Date(Date.now() - (100 - i) * 24 * 60 * 60 * 1000).toISOString(),
            value: prevValue + change
          }];
        }, []),
        monthlyReturns: Array(12).fill().map((_, i) => ({
          month: new Date(2023, i, 1).toLocaleString('default', { month: 'long' }),
          return: (Math.random() * 2 - 0.5) * 10
        }))
      };
      
      // Update strategy with backtest results
      setStrategy({
        ...strategy,
        backtestResults: results
      });
      
      setBacktestResults(results);
      setBacktestInProgress(false);
      setSuccess('Backtest completed successfully');
      setTimeout(() => setSuccess(null), 3000);
    }, 3000);
  };

  const handleSaveStrategy = () => {
    if (!strategy.name) {
      setError('Strategy name is required');
      return;
    }
    
    // Update timestamps
    const updatedStrategy = {
      ...strategy,
      updatedAt: new Date().toISOString()
    };
    
    // Call the onSave callback
    if (onSave) {
      onSave(updatedStrategy);
    }
    
    setSuccess('Strategy saved successfully');
    setTimeout(() => setSuccess(null), 3000);
  };

  const renderPreviewChart = () => {
    if (!previewData) return <div className="text-center p-5"><Spinner animation="border" /></div>;

    const chartData = {
      labels: previewData.dates,
      datasets: [
        {
          label: 'Price',
          data: previewData.prices,
          borderColor: 'rgba(75, 192, 192, 1)',
          backgroundColor: 'rgba(75, 192, 192, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 5,
          fill: true,
        }
      ]
    };

    // Add indicator lines based on strategy
    if (strategy.indicators.some(ind => ind.id === 'sma')) {
      const smaIndicator = strategy.indicators.find(ind => ind.id === 'sma');
      const period = smaIndicator?.parameters?.period || 20;
      
      if (period === 20 && previewData.indicators.sma20) {
        chartData.datasets.push({
          label: `SMA (${period})`,
          data: previewData.indicators.sma20,
          borderColor: 'rgba(255, 99, 132, 1)',
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
        });
      } else if (period === 50 && previewData.indicators.sma50) {
        chartData.datasets.push({
          label: `SMA (${period})`,
          data: previewData.indicators.sma50,
          borderColor: 'rgba(255, 159, 64, 1)',
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
        });
      }
    }

    const options = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: 'Strategy Preview',
        },
        tooltip: {
          mode: 'index',
          intersect: false,
        }
      },
      scales: {
        x: {
          grid: {
            display: true,
            color: 'rgba(200, 200, 200, 0.2)',
          }
        },
        y: {
          grid: {
            display: true,
            color: 'rgba(200, 200, 200, 0.2)',
          }
        }
      },
      interaction: {
        mode: 'nearest',
        axis: 'x',
        intersect: false
      }
    };

    return (
      <div style={{ height: '400px' }}>
        <Line data={chartData} options={options} />
      </div>
    );
  };

  const renderBacktestResults = () => {
    if (!backtestResults) return null;

    const equityCurveData = {
      labels: backtestResults.equityCurve.map(point => new Date(point.date).toLocaleDateString()),
      datasets: [
        {
          label: 'Equity Curve',
          data: backtestResults.equityCurve.map(point => point.value),
          borderColor: 'rgba(75, 192, 192, 1)',
          backgroundColor: 'rgba(75, 192, 192, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
        }
      ]
    };

    const monthlyReturnsData = {
      labels: backtestResults.monthlyReturns.map(m => m.month),
      datasets: [
        {
          label: 'Monthly Returns (%)',
          data: backtestResults.monthlyReturns.map(m => m.return),
          backgroundColor: backtestResults.monthlyReturns.map(m => 
            m.return >= 0 ? 'rgba(75, 192, 192, 0.7)' : 'rgba(255, 99, 132, 0.7)'
          ),
          borderWidth: 1,
        }
      ]
    };

    return (
      <div className="backtest-results mt-4">
        <h4>Backtest Results</h4>
        
        <Row className="mt-3">
          <Col md={6}>
            <Card className="mb-3">
              <Card.Body>
                <Card.Title>Performance Summary</Card.Title>
                <Table striped bordered hover size="sm">
                  <tbody>
                    <tr>
                      <td>Total Return</td>
                      <td className={backtestResults.summary.totalReturn >= 0 ? 'text-success' : 'text-danger'}>
                        {backtestResults.summary.totalReturn.toFixed(2)}%
                      </td>
                    </tr>
                    <tr>
                      <td>Annualized Return</td>
                      <td className={backtestResults.summary.annualizedReturn >= 0 ? 'text-success' : 'text-danger'}>
                        {backtestResults.summary.annualizedReturn.toFixed(2)}%
                      </td>
                    </tr>
                    <tr>
                      <td>Sharpe Ratio</td>
                      <td>{backtestResults.summary.sharpeRatio.toFixed(2)}</td>
                    </tr>
                    <tr>
                      <td>Max Drawdown</td>
                      <td className="text-danger">{backtestResults.summary.maxDrawdown.toFixed(2)}%</td>
                    </tr>
                    <tr>
                      <td>Win Rate</td>
                      <td>{backtestResults.summary.winRate.toFixed(2)}%</td>
                    </tr>
                    <tr>
                      <td>Profit Factor</td>
                      <td>{backtestResults.summary.profitFactor.toFixed(2)}</td>
                    </tr>
                    <tr>
                      <td>Total Trades</td>
                      <td>{backtestResults.summary.totalTrades}</td>
                    </tr>
                  </tbody>
                </Table>
              </Card.Body>
            </Card>
          </Col>
          
          <Col md={6}>
            <Card className="mb-3">
              <Card.Body>
                <Card.Title>Equity Curve</Card.Title>
                <div style={{ height: '200px' }}>
                  <Line 
                    data={equityCurveData} 
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          display: false,
                        }
                      },
                      scales: {
                        x: {
                          display: false
                        }
                      }
                    }} 
                  />
                </div>
              </Card.Body>
            </Card>
          </Col>
        </Row>
        
        <Row>
          <Col md={12}>
            <Card className="mb-3">
              <Card.Body>
                <Card.Title>Trade History</Card.Title>
                <div className="table-responsive">
                  <Table striped bordered hover size="sm">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Symbol</th>
                        <th>Entry Date</th>
                        <th>Exit Date</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>Quantity</th>
                        <th>P&L</th>
                        <th>P&L %</th>
                        <th>Holding Period</th>
                      </tr>
                    </thead>
                    <tbody>
                      {backtestResults.trades.slice(0, 10).map(trade => (
                        <tr key={trade.id}>
                          <td>{trade.id}</td>
                          <td>{trade.symbol}</td>
                          <td>{new Date(trade.entryDate).toLocaleDateString()}</td>
                          <td>{new Date(trade.exitDate).toLocaleDateString()}</td>
                          <td>${trade.entryPrice.toFixed(2)}</td>
                          <td>${trade.exitPrice.toFixed(2)}</td>
                          <td>{trade.quantity}</td>
                          <td className={trade.pnl >= 0 ? 'text-success' : 'text-danger'}>
                            ${trade.pnl.toFixed(2)}
                          </td>
                          <td className={trade.pnlPercent >= 0 ? 'text-success' : 'text-danger'}>
                            {trade.pnlPercent.toFixed(2)}%
                          </td>
                          <td>{trade.holdingPeriod} days</td>
                        </tr>
                      ))}
                    </tbody>
                  </Table>
                  {backtestResults.trades.length > 10 && (
                    <div className="text-center mt-2">
                      <small>Showing 10 of {backtestResults.trades.length} trades</small>
                    </div>
                  )}
                </div>
              </Card.Body>
            </Card>
          </Col>
        </Row>
      </div>
    );
  };

  return (
    <Container fluid className="strategy-builder">
      <Row className="mb-3">
        <Col>
          <h2>{existingStrategy ? 'Edit Strategy' : 'Create New Strategy'}</h2>
          {error && <Alert variant="danger">{error}</Alert>}
          {success && <Alert variant="success">{success}</Alert>}
        </Col>
      </Row>

      <Row>
        <Col md={8}>
          <Card>
            <Card.Body>
              <Tabs
                activeKey={activeTab}
                onSelect={(k) => setActiveTab(k)}
                className="mb-3"
              >
                <Tab eventKey="general" title="General">
                  <Form>
                    <Form.Group className="mb-3">
                      <Form.Label>Strategy Name</Form.Label>
                      <Form.Control
                        type="text"
                        name="name"
                        value={strategy.name}
                        onChange={handleInputChange}
                        placeholder="Enter strategy name"
                      />
                    </Form.Group>

                    <Form.Group className="mb-3">
                      <Form.Label>Description</Form.Label>
                      <Form.Control
                        as="textarea"
                        name="description"
                        value={strategy.description}
                        onChange={handleInputChange}
                        placeholder="Describe your strategy"
                        rows={3}
                      />
                    </Form.Group>

                    <Row>
                      <Col md={6}>
                        <Form.Group className="mb-3">
                          <Form.Label>Strategy Type</Form.Label>
                          <Form.Select
                            name="type"
                            value={strategy.type}
                            onChange={handleInputChange}
                          >
                            <option value="trend_following">Trend Following</option>
                            <option value="mean_reversion">Mean Reversion</option>
                            <option value="breakout">Breakout</option>
                            <option value="momentum">Momentum</option>
                            <option value="volatility">Volatility</option>
                            <option value="pattern">Pattern Recognition</option>
                            <option value="multi_factor">Multi-Factor</option>
                            <option value="custom">Custom</option>
                          </Form.Select>
                        </Form.Group>
                      </Col>
                      <Col md={6}>
                        <Form.Group className="mb-3">
                          <Form.Label>Timeframe</Form.Label>
                          <Form.Select
                            name="timeframe"
                            value={strategy.timeframe}
                            onChange={handleInputChange}
                          >
                            <option value="1m">1 Minute</option>
                            <option value="5m">5 Minutes</option>
                            <option value="15m">15 Minutes</option>
                            <option value="30m">30 Minutes</option>
                            <option value="1h">1 Hour</option>
                            <option value="4h">4 Hours</option>
                            <option value="1d">Daily</option>
                            <option value="1w">Weekly</option>
                          </Form.Select>
                        </Form.Group>
                      </Col>
                    </Row>

                    <Form.Group className="mb-3">
                      <Form.Check
                        type="switch"
                        id="active-switch"
                        label="Active Strategy"
                        name="isActive"
                        checked={strategy.isActive}
                        onChange={handleCheckboxChange}
                      />
                      <Form.Text className="text-muted">
                        When active, this strategy will generate real-time signals
                      </Form.Text>
                    </Form.Group>
                  </Form>
                </Tab>

                <Tab eventKey="indicators" title="Indicators">
                  <div className="mb-4">
                    <h5>Selected Indicators</h5>
                    {strategy.indicators.length === 0 ? (
                      <p className="text-muted">No indicators added yet</p>
                    ) : (
                      <div>
                        {strategy.indicators.map(indicator => (
                          <Card key={indicator.id} className="mb-2">
                            <Card.Body>
                              <div className="d-flex justify-content-between align-items-center">
                                <h6>{indicator.name}</h6>
                                <Button 
                                  variant="outline-danger" 
                                  size="sm"
                                  onClick={() => handleRemoveIndicator(indicator.id)}
                                >
                                  Remove
                                </Button>
                              </div>
                              
                              <div className="mt-2">
                                {Object.keys(indicator.parameters).map(param => (
                                  <Form.Group key={param} className="mb-2">
                                    <Form.Label>{param}</Form.Label>
                                    <Form.Control
                                      type="number"
                                      value={indicator.parameters[param]}
                                      onChange={(e) => handleUpdateIndicatorParam(indicator.id, param, e.target.value)}
                                      size="sm"
                                    />
                                  </Form.Group>
                                ))}
                              </div>
                            </Card.Body>
                          </Card>
                        ))}
                      </div>
                    )}
                  </div>

                  <div>
                    <h5>Available Indicators</h5>
                    <Row>
                      {availableIndicators.map(indicator => (
                        <Col md={4} key={indicator.id} className="mb-2">
                          <Button
                            variant="outline-primary"
                            size="sm"
                            className="w-100"
                            onClick={() => handleAddIndicator(indicator)}
                          >
                            {indicator.name}
                          </Button>
                        </Col>
                      ))}
                    </Row>
                  </div>
                </Tab>

                <Tab eventKey="conditions" title="Conditions">
                  <Tabs
                    activeKey={conditionType}
                    onSelect={(k) => setConditionType(k)}
                    className="mb-3"
                  >
                    <Tab eventKey="entry" title="Entry Conditions">
                      <div className="mb-4">
                        <h5>Entry Conditions</h5>
                        {strategy.entryConditions.length === 0 ? (
                          <p className="text-muted">No entry conditions defined yet</p>
                        ) : (
                          <div>
                            {strategy.entryConditions.map(condition => (
                              <Card key={condition.id} className="mb-2">
                                <Card.Body>
                                  <div className="d-flex justify-content-between align-items-center">
                                    <div>
                                      <strong>{condition.indicator}</strong>
                                      {condition.lookback > 1 && ` [${condition.lookback}]`}
                                      {' '}{condition.comparator}{' '}
                                      {condition.useSecondaryIndicator ? 
                                        condition.secondaryIndicator : 
                                        condition.value
                                      }
                                    </div>
                                    <Button 
                                      variant="outline-danger" 
                                      size="sm"
                                      onClick={() => handleRemoveCondition(condition.id, 'entry')}
                                    >
                                      Remove
                                    </Button>
                                  </div>
                                </Card.Body>
                              </Card>
                            ))}
                          </div>
                        )}
                      </div>

                      <div className="mt-3">
                        <h5>Add Entry Condition</h5>
                        <Form>
                          <Row className="mb-3">
                            <Col md={5}>
                              <Form.Group>
                                <Form.Label>Indicator</Form.Label>
                                <Form.Select
                                  name="indicator"
                                  value={newCondition.indicator}
                                  onChange={handleNewConditionChange}
                                >
                                  <option value="">Select Indicator</option>
                                  {strategy.indicators.map(ind => (
                                    <option key={ind.id} value={ind.id}>{ind.name}</option>
                                  ))}
                                  <option value="price">Price</option>
                                  <option value="volume">Volume</option>
                                </Form.Select>
                              </Form.Group>
                            </Col>
                            <Col md={3}>
                              <Form.Group>
                                <Form.Label>Comparator</Form.Label>
                                <Form.Select
                                  name="comparator"
                                  value={newCondition.comparator}
                                  onChange={handleNewConditionChange}
                                >
                                  <option value=">">Greater Than (&gt;)</option>
                                  <option value="<">Less Than (&lt;)</option>
                                  <option value=">=">Greater Than or Equal (&ge;)</option>
                                  <option value="<=">Less Than or Equal (&le;)</option>
                                  <option value="==">Equal To (=)</option>
                                  <option value="crosses_above">Crosses Above</option>
                                  <option value="crosses_below">Crosses Below</option>
                                </Form.Select>
                              </Form.Group>
                            </Col>
                            <Col md={4}>
                              <Form.Group>
                                <Form.Label>Value</Form.Label>
                                <Form.Control
                                  type="text"
                                  name="value"
                                  value={newCondition.value}
                                  onChange={handleNewConditionChange}
                                  placeholder="Value or indicator"
                                />
                              </Form.Group>
                            </Col>
                          </Row>

                          <Row className="mb-3">
                            <Col md={4}>
                              <Form.Group>
                                <Form.Label>Lookback Periods</Form.Label>
                                <Form.Control
                                  type="number"
                                  name="lookback"
                                  value={newCondition.lookback}
                                  onChange={handleNewConditionChange}
                                  min="1"
                                />
                              </Form.Group>
                            </Col>
                            <Col md={8}>
                              <Form.Group className="mt-4">
                                <Form.Check
                                  type="checkbox"
                                  id="use-secondary-indicator"
                                  label="Compare to another indicator"
                                  name="useSecondaryIndicator"
                                  checked={newCondition.useSecondaryIndicator}
                                  onChange={(e) => setNewCondition({
                                    ...newCondition,
                                    useSecondaryIndicator: e.target.checked
                                  })}
                                />
                              </Form.Group>
                            </Col>
                          </Row>

                          {newCondition.useSecondaryIndicator && (
                            <Row className="mb-3">
                              <Col md={8}>
                                <Form.Group>
                                  <Form.Label>Secondary Indicator</Form.Label>
                                  <Form.Select
                                    name="secondaryIndicator"
                                    value={newCondition.secondaryIndicator}
                                    onChange={handleNewConditionChange}
                                  >
                                    <option value="">Select Indicator</option>
                                    {strategy.indicators.map(ind => (
                                      <option key={ind.id} value={ind.id}>{ind.name}</option>
                                    ))}
                                    <option value="price">Price</option>
                                    <option value="volume">Volume</option>
                                  </Form.Select>
                                </Form.Group>
                              </Col>
                              <Col md={4}>
                                <Form.Group>
                                  <Form.Label>Lookback</Form.Label>
                                  <Form.Control
                                    type="number"
                                    name="secondaryLookback"
                                    value={newCondition.secondaryLookback || 1}
                                    onChange={handleNewConditionChange}
                                    min="1"
                                  />
                                </Form.Group>
                              </Col>
                            </Row>
                          )}

                          <Button 
                            variant="primary" 
                            onClick={handleAddCondition}
                          >
                            Add Condition
                          </Button>
                        </Form>
                      </div>
                    </Tab>

                    <Tab eventKey="exit" title="Exit Conditions">
                      <div className="mb-4">
                        <h5>Exit Conditions</h5>
                        {strategy.exitConditions.length === 0 ? (
                          <p className="text-muted">No exit conditions defined yet</p>
                        ) : (
                          <div>
                            {strategy.exitConditions.map(condition => (
                              <Card key={condition.id} className="mb-2">
                                <Card.Body>
                                  <div className="d-flex justify-content-between align-items-center">
                                    <div>
                                      <strong>{condition.indicator}</strong>
                                      {condition.lookback > 1 && ` [${condition.lookback}]`}
                                      {' '}{condition.comparator}{' '}
                                      {condition.useSecondaryIndicator ? 
                                        condition.secondaryIndicator : 
                                        condition.value
                                      }
                                    </div>
                                    <Button 
                                      variant="outline-danger" 
                                      size="sm"
                                      onClick={() => handleRemoveCondition(condition.id, 'exit')}
                                    >
                                      Remove
                                    </Button>
                                  </div>
                                </Card.Body>
                              </Card>
                            ))}
                          </div>
                        )}
                      </div>

                      <div className="mt-3">
                        <h5>Add Exit Condition</h5>
                        <Form>
                          <Row className="mb-3">
                            <Col md={5}>
                              <Form.Group>
                                <Form.Label>Indicator</Form.Label>
                                <Form.Select
                                  name="indicator"
                                  value={newCondition.indicator}
                                  onChange={handleNewConditionChange}
                                >
                                  <option value="">Select Indicator</option>
                                  {strategy.indicators.map(ind => (
                                    <option key={ind.id} value={ind.id}>{ind.name}</option>
                                  ))}
                                  <option value="price">Price</option>
                                  <option value="volume">Volume</option>
                                </Form.Select>
                              </Form.Group>
                            </Col>
                            <Col md={3}>
                              <Form.Group>
                                <Form.Label>Comparator</Form.Label>
                                <Form.Select
                                  name="comparator"
                                  value={newCondition.comparator}
                                  onChange={handleNewConditionChange}
                                >
                                  <option value=">">Greater Than (&gt;)</option>
                                  <option value="<">Less Than (&lt;)</option>
                                  <option value=">=">Greater Than or Equal (&ge;)</option>
                                  <option value="<=">Less Than or Equal (&le;)</option>
                                  <option value="==">Equal To (=)</option>
                                  <option value="crosses_above">Crosses Above</option>
                                  <option value="crosses_below">Crosses Below</option>
                                </Form.Select>
                              </Form.Group>
                            </Col>
                            <Col md={4}>
                              <Form.Group>
                                <Form.Label>Value</Form.Label>
                                <Form.Control
                                  type="text"
                                  name="value"
                                  value={newCondition.value}
                                  onChange={handleNewConditionChange}
                                  placeholder="Value or indicator"
                                />
                              </Form.Group>
                            </Col>
                          </Row>

                          <Button 
                            variant="primary" 
                            onClick={handleAddCondition}
                          >
                            Add Condition
                          </Button>
                        </Form>
                      </div>
                    </Tab>
                  </Tabs>
                </Tab>

                <Tab eventKey="risk" title="Risk Management">
                  <Form>
                    <Row>
                      <Col md={6}>
                        <Form.Group className="mb-3">
                          <Form.Label>Position Size (% of portfolio)</Form.Label>
                          <Form.Control
                            type="number"
                            name="riskManagement.positionSize"
                            value={strategy.riskManagement.positionSize}
                            onChange={handleNumberInputChange}
                            min="0.1"
                            max="100"
                            step="0.1"
                          />
                          <Form.Text className="text-muted">
                            Percentage of portfolio to allocate to each position
                          </Form.Text>
                        </Form.Group>
                      </Col>
                      <Col md={6}>
                        <Form.Group className="mb-3">
                          <Form.Label>Maximum Open Positions</Form.Label>
                          <Form.Control
                            type="number"
                            name="riskManagement.maxOpenPositions"
                            value={strategy.riskManagement.maxOpenPositions}
                            onChange={handleNumberInputChange}
                            min="1"
                            max="100"
                          />
                          <Form.Text className="text-muted">
                            Maximum number of positions to hold simultaneously
                          </Form.Text>
                        </Form.Group>
                      </Col>
                    </Row>

                    <Row>
                      <Col md={6}>
                        <Form.Group className="mb-3">
                          <Form.Label>Stop Loss (%)</Form.Label>
                          <Form.Control
                            type="number"
                            name="riskManagement.stopLoss"
                            value={strategy.riskManagement.stopLoss}
                            onChange={handleNumberInputChange}
                            min="0.1"
                            max="100"
                            step="0.1"
                          />
                          <Form.Text className="text-muted">
                            Percentage below entry price to set stop loss
                          </Form.Text>
                        </Form.Group>
                      </Col>
                      <Col md={6}>
                        <Form.Group className="mb-3">
                          <Form.Label>Take Profit (%)</Form.Label>
                          <Form.Control
                            type="number"
                            name="riskManagement.takeProfit"
                            value={strategy.riskManagement.takeProfit}
                            onChange={handleNumberInputChange}
                            min="0.1"
                            max="1000"
                            step="0.1"
                          />
                          <Form.Text className="text-muted">
                            Percentage above entry price to set take profit
                          </Form.Text>
                        </Form.Group>
                      </Col>
                    </Row>

                    <Form.Group className="mb-3">
                      <Form.Check
                        type="switch"
                        id="trailing-stop-switch"
                        label="Use Trailing Stop"
                        name="riskManagement.trailingStop"
                        checked={strategy.riskManagement.trailingStop}
                        onChange={handleCheckboxChange}
                      />
                      <Form.Text className="text-muted">
                        Trailing stop follows price movement to lock in profits
                      </Form.Text>
                    </Form.Group>

                    {strategy.riskManagement.trailingStop && (
                      <Form.Group className="mb-3">
                        <Form.Label>Trailing Stop Distance (%)</Form.Label>
                        <Form.Control
                          type="number"
                          name="riskManagement.trailingStopDistance"
                          value={strategy.riskManagement.trailingStopDistance}
                          onChange={handleNumberInputChange}
                          min="0.1"
                          max="100"
                          step="0.1"
                        />
                        <Form.Text className="text-muted">
                          Distance between current price and trailing stop
                        </Form.Text>
                      </Form.Group>
                    )}
                  </Form>
                </Tab>

                <Tab eventKey="filters" title="Filters">
                  <Form>
                    <Row>
                      <Col md={6}>
                        <Form.Group className="mb-3">
                          <Form.Label>Minimum Price ($)</Form.Label>
                          <Form.Control
                            type="number"
                            name="filters.minPrice"
                            value={strategy.filters.minPrice}
                            onChange={handleNumberInputChange}
                            min="0"
                            step="0.01"
                          />
                        </Form.Group>
                      </Col>
                      <Col md={6}>
                        <Form.Group className="mb-3">
                          <Form.Label>Maximum Price ($)</Form.Label>
                          <Form.Control
                            type="number"
                            name="filters.maxPrice"
                            value={strategy.filters.maxPrice}
                            onChange={handleNumberInputChange}
                            min="0"
                            step="0.01"
                          />
                        </Form.Group>
                      </Col>
                    </Row>

                    <Form.Group className="mb-3">
                      <Form.Label>Minimum Volume</Form.Label>
                      <Form.Control
                        type="number"
                        name="filters.minVolume"
                        value={strategy.filters.minVolume}
                        onChange={handleNumberInputChange}
                        min="0"
                      />
                      <Form.Text className="text-muted">
                        Minimum average daily trading volume
                      </Form.Text>
                    </Form.Group>

                    <Form.Group className="mb-3">
                      <Form.Label>Sectors</Form.Label>
                      <div>
                        {['Technology', 'Healthcare', 'Financials', 'Consumer Discretionary', 
                          'Consumer Staples', 'Industrials', 'Energy', 'Materials', 
                          'Utilities', 'Real Estate', 'Communication Services'].map(sector => (
                          <Form.Check
                            key={sector}
                            type="checkbox"
                            id={`sector-${sector}`}
                            label={sector}
                            checked={strategy.filters.sectors.includes(sector)}
                            onChange={(e) => handleSectorChange(sector, e.target.checked)}
                            inline
                          />
                        ))}
                      </div>
                    </Form.Group>

                    <Form.Group className="mb-3">
                      <Form.Label>Excluded Symbols</Form.Label>
                      <div className="d-flex">
                        <Form.Control
                          type="text"
                          value={newFilter}
                          onChange={(e) => setNewFilter(e.target.value)}
                          placeholder="Enter symbol to exclude"
                          className="me-2"
                        />
                        <Button 
                          variant="outline-primary"
                          onClick={handleAddExcludedSymbol}
                        >
                          Add
                        </Button>
                      </div>
                      <div className="mt-2">
                        {strategy.filters.excludedSymbols.map(symbol => (
                          <Badge 
                            key={symbol} 
                            bg="secondary" 
                            className="me-2 mb-2"
                            style={{ cursor: 'pointer' }}
                            onClick={() => handleRemoveExcludedSymbol(symbol)}
                          >
                            {symbol} &times;
                          </Badge>
                        ))}
                      </div>
                    </Form.Group>
                  </Form>
                </Tab>

                <Tab eventKey="backtest" title="Backtest">
                  <div className="mb-3">
                    <p>Run a backtest to evaluate your strategy's performance.</p>
                    <Button 
                      variant="primary" 
                      onClick={handleRunBacktest}
                      disabled={backtestInProgress}
                    >
                      {backtestInProgress ? (
                        <>
                          <Spinner
                            as="span"
                            animation="border"
                            size="sm"
                            role="status"
                            aria-hidden="true"
                            className="me-2"
                          />
                          Running Backtest...
                        </>
                      ) : 'Run Backtest'}
                    </Button>
                  </div>

                  {renderBacktestResults()}
                </Tab>
              </Tabs>
            </Card.Body>
          </Card>
        </Col>

        <Col md={4}>
          <Card className="mb-3">
            <Card.Body>
              <Card.Title>Strategy Preview</Card.Title>
              {renderPreviewChart()}
            </Card.Body>
          </Card>

          <Card>
            <Card.Body>
              <Card.Title>Strategy Summary</Card.Title>
              <Table striped bordered hover size="sm">
                <tbody>
                  <tr>
                    <td>Name</td>
                    <td>{strategy.name || 'Unnamed Strategy'}</td>
                  </tr>
                  <tr>
                    <td>Type</td>
                    <td>{strategy.type.replace('_', ' ').toUpperCase()}</td>
                  </tr>
                  <tr>
                    <td>Timeframe</td>
                    <td>{strategy.timeframe}</td>
                  </tr>
                  <tr>
                    <td>Indicators</td>
                    <td>{strategy.indicators.length}</td>
                  </tr>
                  <tr>
                    <td>Entry Conditions</td>
                    <td>{strategy.entryConditions.length}</td>
                  </tr>
                  <tr>
                    <td>Exit Conditions</td>
                    <td>{strategy.exitConditions.length}</td>
                  </tr>
                  <tr>
                    <td>Position Size</td>
                    <td>{strategy.riskManagement.positionSize}%</td>
                  </tr>
                  <tr>
                    <td>Status</td>
                    <td>
                      <Badge bg={strategy.isActive ? 'success' : 'secondary'}>
                        {strategy.isActive ? 'Active' : 'Inactive'}
                      </Badge>
                    </td>
                  </tr>
                </tbody>
              </Table>

              <div className="d-grid gap-2 mt-3">
                <Button 
                  variant="success" 
                  onClick={handleSaveStrategy}
                >
                  Save Strategy
                </Button>
              </div>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default StrategyBuilder;
