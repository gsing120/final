import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Button, Table, Badge, Spinner, Alert, Tabs, Tab, Form, InputGroup } from 'react-bootstrap';
import { Line, Bar, Pie } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, ArcElement, Title, Tooltip, Legend } from 'chart.js';

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, ArcElement, Title, Tooltip, Legend);

const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [portfolioData, setPortfolioData] = useState(null);
  const [activeStrategies, setActiveStrategies] = useState([]);
  const [recentTrades, setRecentTrades] = useState([]);
  const [marketOverview, setMarketOverview] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [watchlist, setWatchlist] = useState([]);
  const [newWatchlistSymbol, setNewWatchlistSymbol] = useState('');
  const [timeframe, setTimeframe] = useState('1d');
  const [refreshInterval, setRefreshInterval] = useState(60000); // 1 minute
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Initial data load
    fetchDashboardData();

    // Set up auto-refresh if enabled
    let intervalId = null;
    if (autoRefresh) {
      intervalId = setInterval(() => {
        fetchDashboardData();
      }, refreshInterval);
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [autoRefresh, refreshInterval, timeframe]);

  const fetchDashboardData = async () => {
    setLoading(true);
    setError(null);

    try {
      // In a real implementation, these would be API calls to the backend
      // Simulating API calls with setTimeout
      setTimeout(() => {
        // Mock portfolio data
        const portfolioData = {
          totalValue: 125750.42,
          cashBalance: 45230.18,
          investedValue: 80520.24,
          dailyChange: 1250.75,
          dailyChangePercent: 1.01,
          totalReturn: 25750.42,
          totalReturnPercent: 25.75,
          allocation: {
            stocks: 65,
            options: 15,
            cash: 20
          },
          history: Array(30).fill().map((_, i) => ({
            date: new Date(Date.now() - (30 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
            value: 100000 + Math.random() * 30000 + i * 1000
          }))
        };
        setPortfolioData(portfolioData);

        // Mock active strategies
        const strategies = [
          {
            id: 1,
            name: 'Momentum Breakout',
            type: 'momentum',
            status: 'active',
            positions: 3,
            dailyPnL: 450.25,
            totalPnL: 2340.50,
            winRate: 68.5
          },
          {
            id: 2,
            name: 'Volatility Mean Reversion',
            type: 'mean_reversion',
            status: 'active',
            positions: 2,
            dailyPnL: -120.75,
            totalPnL: 1560.30,
            winRate: 62.1
          },
          {
            id: 3,
            name: 'Trend Following ETF',
            type: 'trend_following',
            status: 'active',
            positions: 4,
            dailyPnL: 325.50,
            totalPnL: 4250.80,
            winRate: 71.2
          }
        ];
        setActiveStrategies(strategies);

        // Mock recent trades
        const trades = [
          {
            id: 1001,
            symbol: 'AAPL',
            type: 'buy',
            quantity: 25,
            price: 178.25,
            timestamp: new Date(Date.now() - 35 * 60 * 1000).toISOString(),
            strategy: 'Momentum Breakout',
            status: 'filled'
          },
          {
            id: 1002,
            symbol: 'MSFT',
            type: 'sell',
            quantity: 15,
            price: 415.75,
            timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
            strategy: 'Volatility Mean Reversion',
            status: 'filled'
          },
          {
            id: 1003,
            symbol: 'NVDA',
            type: 'buy',
            quantity: 10,
            price: 925.50,
            timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(),
            strategy: 'Trend Following ETF',
            status: 'filled'
          },
          {
            id: 1004,
            symbol: 'AMZN',
            type: 'buy',
            quantity: 12,
            price: 182.30,
            timestamp: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
            strategy: 'Momentum Breakout',
            status: 'filled'
          },
          {
            id: 1005,
            symbol: 'GOOGL',
            type: 'sell',
            quantity: 8,
            price: 175.40,
            timestamp: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
            strategy: 'Volatility Mean Reversion',
            status: 'filled'
          }
        ];
        setRecentTrades(trades);

        // Mock market overview
        const marketData = {
          indices: [
            { name: 'S&P 500', value: 5320.75, change: 0.85, changePercent: 0.85 },
            { name: 'Nasdaq', value: 16750.25, change: 1.25, changePercent: 1.25 },
            { name: 'Dow Jones', value: 39250.50, change: 0.45, changePercent: 0.45 },
            { name: 'Russell 2000', value: 2150.30, change: -0.35, changePercent: -0.35 }
          ],
          sectors: [
            { name: 'Technology', performance: 1.75 },
            { name: 'Healthcare', performance: 0.85 },
            { name: 'Financials', performance: 0.25 },
            { name: 'Consumer Discretionary', performance: 1.15 },
            { name: 'Energy', performance: -0.65 },
            { name: 'Utilities', performance: -0.35 },
            { name: 'Materials', performance: 0.45 }
          ],
          volatility: {
            vix: 15.75,
            vixChange: -0.85,
            historicalVolatility: 12.5
          },
          marketBreadth: {
            advancers: 65,
            decliners: 35,
            newHighs: 125,
            newLows: 45
          }
        };
        setMarketOverview(marketData);

        // Mock alerts
        const alertsData = [
          {
            id: 1,
            type: 'signal',
            priority: 'high',
            message: 'Buy signal triggered for AAPL by Momentum Breakout strategy',
            timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString()
          },
          {
            id: 2,
            type: 'risk',
            priority: 'medium',
            message: 'Portfolio exposure to Technology sector exceeds 40%',
            timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString()
          },
          {
            id: 3,
            type: 'market',
            priority: 'low',
            message: 'VIX dropped below 16, market volatility decreasing',
            timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString()
          }
        ];
        setAlerts(alertsData);

        // Mock watchlist
        const watchlistData = [
          {
            symbol: 'AAPL',
            name: 'Apple Inc.',
            price: 178.25,
            change: 2.35,
            changePercent: 1.34,
            volume: 45250000
          },
          {
            symbol: 'MSFT',
            name: 'Microsoft Corporation',
            price: 415.75,
            change: 5.25,
            changePercent: 1.28,
            volume: 22150000
          },
          {
            symbol: 'NVDA',
            name: 'NVIDIA Corporation',
            price: 925.50,
            change: 15.75,
            changePercent: 1.73,
            volume: 35750000
          },
          {
            symbol: 'AMZN',
            name: 'Amazon.com, Inc.',
            price: 182.30,
            change: -0.85,
            changePercent: -0.46,
            volume: 28450000
          },
          {
            symbol: 'GOOGL',
            name: 'Alphabet Inc.',
            price: 175.40,
            change: 1.25,
            changePercent: 0.72,
            volume: 18250000
          }
        ];
        setWatchlist(watchlistData);

        setLoading(false);
      }, 1000);
    } catch (error) {
      setError('Failed to fetch dashboard data. Please try again later.');
      setLoading(false);
    }
  };

  const handleRefreshData = () => {
    fetchDashboardData();
  };

  const handleTimeframeChange = (newTimeframe) => {
    setTimeframe(newTimeframe);
  };

  const handleAutoRefreshChange = (e) => {
    setAutoRefresh(e.target.checked);
  };

  const handleRefreshIntervalChange = (e) => {
    setRefreshInterval(parseInt(e.target.value, 10));
  };

  const handleAddToWatchlist = () => {
    if (newWatchlistSymbol && !watchlist.some(item => item.symbol === newWatchlistSymbol.toUpperCase())) {
      // In a real implementation, this would fetch data for the new symbol
      // For now, we'll add a mock entry
      const newSymbol = {
        symbol: newWatchlistSymbol.toUpperCase(),
        name: `${newWatchlistSymbol.toUpperCase()} Corp.`,
        price: (Math.random() * 500 + 50).toFixed(2),
        change: (Math.random() * 10 - 5).toFixed(2),
        changePercent: (Math.random() * 5 - 2.5).toFixed(2),
        volume: Math.floor(Math.random() * 50000000)
      };
      
      setWatchlist([...watchlist, newSymbol]);
      setNewWatchlistSymbol('');
    }
  };

  const handleRemoveFromWatchlist = (symbol) => {
    setWatchlist(watchlist.filter(item => item.symbol !== symbol));
  };

  const handleDismissAlert = (id) => {
    setAlerts(alerts.filter(alert => alert.id !== id));
  };

  const renderPortfolioChart = () => {
    if (!portfolioData) return null;

    const chartData = {
      labels: portfolioData.history.map(point => point.date),
      datasets: [
        {
          label: 'Portfolio Value',
          data: portfolioData.history.map(point => point.value),
          borderColor: 'rgba(75, 192, 192, 1)',
          backgroundColor: 'rgba(75, 192, 192, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 5,
          fill: true,
        }
      ]
    };

    const options = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          mode: 'index',
          intersect: false,
        }
      },
      scales: {
        x: {
          grid: {
            display: false,
          }
        },
        y: {
          grid: {
            color: 'rgba(200, 200, 200, 0.2)',
          },
          ticks: {
            callback: function(value) {
              return '$' + value.toLocaleString();
            }
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
      <div style={{ height: '300px' }}>
        <Line data={chartData} options={options} />
      </div>
    );
  };

  const renderAllocationChart = () => {
    if (!portfolioData) return null;

    const chartData = {
      labels: Object.keys(portfolioData.allocation).map(key => key.charAt(0).toUpperCase() + key.slice(1)),
      datasets: [
        {
          data: Object.values(portfolioData.allocation),
          backgroundColor: [
            'rgba(54, 162, 235, 0.7)',
            'rgba(255, 99, 132, 0.7)',
            'rgba(255, 206, 86, 0.7)',
          ],
          borderColor: [
            'rgba(54, 162, 235, 1)',
            'rgba(255, 99, 132, 1)',
            'rgba(255, 206, 86, 1)',
          ],
          borderWidth: 1,
        }
      ]
    };

    const options = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'right',
        }
      }
    };

    return (
      <div style={{ height: '200px' }}>
        <Pie data={chartData} options={options} />
      </div>
    );
  };

  const renderSectorPerformance = () => {
    if (!marketOverview) return null;

    const chartData = {
      labels: marketOverview.sectors.map(sector => sector.name),
      datasets: [
        {
          label: 'Performance (%)',
          data: marketOverview.sectors.map(sector => sector.performance),
          backgroundColor: marketOverview.sectors.map(sector => 
            sector.performance >= 0 ? 'rgba(75, 192, 192, 0.7)' : 'rgba(255, 99, 132, 0.7)'
          ),
          borderColor: marketOverview.sectors.map(sector => 
            sector.performance >= 0 ? 'rgba(75, 192, 192, 1)' : 'rgba(255, 99, 132, 1)'
          ),
          borderWidth: 1,
        }
      ]
    };

    const options = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false,
        }
      },
      scales: {
        y: {
          grid: {
            color: 'rgba(200, 200, 200, 0.2)',
          }
        }
      }
    };

    return (
      <div style={{ height: '200px' }}>
        <Bar data={chartData} options={options} />
      </div>
    );
  };

  return (
    <Container fluid className="dashboard">
      <Row className="mb-3">
        <Col>
          <div className="d-flex justify-content-between align-items-center">
            <h2>Dashboard</h2>
            <div>
              <Button 
                variant="outline-primary" 
                size="sm" 
                onClick={handleRefreshData}
                disabled={loading}
                className="me-2"
              >
                {loading ? <Spinner animation="border" size="sm" /> : 'Refresh'}
              </Button>
              <Form.Check
                type="switch"
                id="auto-refresh-switch"
                label="Auto Refresh"
                checked={autoRefresh}
                onChange={handleAutoRefreshChange}
                inline
                className="me-2"
              />
              <Form.Select
                size="sm"
                value={refreshInterval}
                onChange={handleRefreshIntervalChange}
                style={{ width: '120px', display: 'inline-block' }}
                disabled={!autoRefresh}
                className="me-2"
              >
                <option value="30000">30 seconds</option>
                <option value="60000">1 minute</option>
                <option value="300000">5 minutes</option>
                <option value="600000">10 minutes</option>
              </Form.Select>
              <Button variant="outline-secondary" size="sm" className="me-2" onClick={() => handleTimeframeChange('1d')}>1D</Button>
              <Button variant="outline-secondary" size="sm" className="me-2" onClick={() => handleTimeframeChange('1w')}>1W</Button>
              <Button variant="outline-secondary" size="sm" className="me-2" onClick={() => handleTimeframeChange('1m')}>1M</Button>
              <Button variant="outline-secondary" size="sm" onClick={() => handleTimeframeChange('1y')}>1Y</Button>
            </div>
          </div>
        </Col>
      </Row>

      {error && (
        <Row className="mb-3">
          <Col>
            <Alert variant="danger">{error}</Alert>
          </Col>
        </Row>
      )}

      <Row className="mb-4">
        <Col lg={8}>
          <Card className="h-100">
            <Card.Body>
              <Card.Title>Portfolio Performance</Card.Title>
              {loading ? (
                <div className="text-center p-5"><Spinner animation="border" /></div>
              ) : (
                <>
                  {renderPortfolioChart()}
                  <Row className="mt-3">
                    <Col md={3} className="text-center">
                      <div className="text-muted">Total Value</div>
                      <h4>${portfolioData?.totalValue.toLocaleString()}</h4>
                    </Col>
                    <Col md={3} className="text-center">
                      <div className="text-muted">Daily Change</div>
                      <h4 className={portfolioData?.dailyChange >= 0 ? 'text-success' : 'text-danger'}>
                        {portfolioData?.dailyChange >= 0 ? '+' : ''}${portfolioData?.dailyChange.toLocaleString()} ({portfolioData?.dailyChangePercent}%)
                      </h4>
                    </Col>
                    <Col md={3} className="text-center">
                      <div className="text-muted">Cash Balance</div>
                      <h4>${portfolioData?.cashBalance.toLocaleString()}</h4>
                    </Col>
                    <Col md={3} className="text-center">
                      <div className="text-muted">Total Return</div>
                      <h4 className={portfolioData?.totalReturn >= 0 ? 'text-success' : 'text-danger'}>
                        {portfolioData?.totalReturn >= 0 ? '+' : ''}${portfolioData?.totalReturn.toLocaleString()} ({portfolioData?.totalReturnPercent}%)
                      </h4>
                    </Col>
                  </Row>
                </>
              )}
            </Card.Body>
          </Card>
        </Col>
        <Col lg={4}>
          <Card className="h-100">
            <Card.Body>
              <Card.Title>Portfolio Allocation</Card.Title>
              {loading ? (
                <div className="text-center p-5"><Spinner animation="border" /></div>
              ) : (
                renderAllocationChart()
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row className="mb-4">
        <Col md={6}>
          <Card className="h-100">
            <Card.Body>
              <Card.Title>Active Strategies</Card.Title>
              {loading ? (
                <div className="text-center p-5"><Spinner animation="border" /></div>
              ) : (
                <div className="table-responsive">
                  <Table hover>
                    <thead>
                      <tr>
                        <th>Strategy</th>
                        <th>Type</th>
                        <th>Positions</th>
                        <th>Daily P&L</th>
                        <th>Total P&L</th>
                        <th>Win Rate</th>
                      </tr>
                    </thead>
                    <tbody>
                      {activeStrategies.map(strategy => (
                        <tr key={strategy.id}>
                          <td>
                            {strategy.name}
                            <Badge 
                              bg={strategy.status === 'active' ? 'success' : 'secondary'} 
                              className="ms-2"
                            >
                              {strategy.status}
                            </Badge>
                          </td>
                          <td>{strategy.type}</td>
                          <td>{strategy.positions}</td>
                          <td className={strategy.dailyPnL >= 0 ? 'text-success' : 'text-danger'}>
                            {strategy.dailyPnL >= 0 ? '+' : ''}${strategy.dailyPnL.toFixed(2)}
                          </td>
                          <td className={strategy.totalPnL >= 0 ? 'text-success' : 'text-danger'}>
                            {strategy.totalPnL >= 0 ? '+' : ''}${strategy.totalPnL.toFixed(2)}
                          </td>
                          <td>{strategy.winRate}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </Table>
                </div>
              )}
            </Card.Body>
          </Card>
        </Col>
        <Col md={6}>
          <Card className="h-100">
            <Card.Body>
              <Card.Title>Recent Trades</Card.Title>
              {loading ? (
                <div className="text-center p-5"><Spinner animation="border" /></div>
              ) : (
                <div className="table-responsive">
                  <Table hover>
                    <thead>
                      <tr>
                        <th>Symbol</th>
                        <th>Type</th>
                        <th>Quantity</th>
                        <th>Price</th>
                        <th>Time</th>
                        <th>Strategy</th>
                      </tr>
                    </thead>
                    <tbody>
                      {recentTrades.map(trade => (
                        <tr key={trade.id}>
                          <td>{trade.symbol}</td>
                          <td>
                            <Badge bg={trade.type === 'buy' ? 'success' : 'danger'}>
                              {trade.type.toUpperCase()}
                            </Badge>
                          </td>
                          <td>{trade.quantity}</td>
                          <td>${trade.price.toFixed(2)}</td>
                          <td>{new Date(trade.timestamp).toLocaleTimeString()}</td>
                          <td>{trade.strategy}</td>
                        </tr>
                      ))}
                    </tbody>
                  </Table>
                </div>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row className="mb-4">
        <Col md={4}>
          <Card className="h-100">
            <Card.Body>
              <Card.Title>Market Overview</Card.Title>
              {loading ? (
                <div className="text-center p-5"><Spinner animation="border" /></div>
              ) : (
                <>
                  <h6>Major Indices</h6>
                  <Table hover size="sm">
                    <thead>
                      <tr>
                        <th>Index</th>
                        <th>Value</th>
                        <th>Change</th>
                      </tr>
                    </thead>
                    <tbody>
                      {marketOverview?.indices.map((index, i) => (
                        <tr key={i}>
                          <td>{index.name}</td>
                          <td>{index.value.toLocaleString()}</td>
                          <td className={index.change >= 0 ? 'text-success' : 'text-danger'}>
                            {index.change >= 0 ? '+' : ''}{index.change.toFixed(2)} ({index.changePercent}%)
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </Table>

                  <div className="d-flex justify-content-between mt-3">
                    <div>
                      <h6>VIX</h6>
                      <div className="d-flex align-items-center">
                        <h3>{marketOverview?.volatility.vix.toFixed(2)}</h3>
                        <span className={marketOverview?.volatility.vixChange >= 0 ? 'text-danger ms-2' : 'text-success ms-2'}>
                          {marketOverview?.volatility.vixChange >= 0 ? '+' : ''}{marketOverview?.volatility.vixChange.toFixed(2)}
                        </span>
                      </div>
                    </div>
                    <div>
                      <h6>Market Breadth</h6>
                      <div>Adv/Dec: {marketOverview?.marketBreadth.advancers}/{marketOverview?.marketBreadth.decliners}</div>
                      <div>New Hi/Lo: {marketOverview?.marketBreadth.newHighs}/{marketOverview?.marketBreadth.newLows}</div>
                    </div>
                  </div>
                </>
              )}
            </Card.Body>
          </Card>
        </Col>
        <Col md={4}>
          <Card className="h-100">
            <Card.Body>
              <Card.Title>Sector Performance</Card.Title>
              {loading ? (
                <div className="text-center p-5"><Spinner animation="border" /></div>
              ) : (
                renderSectorPerformance()
              )}
            </Card.Body>
          </Card>
        </Col>
        <Col md={4}>
          <Card className="h-100">
            <Card.Body>
              <Card.Title>Alerts</Card.Title>
              {loading ? (
                <div className="text-center p-5"><Spinner animation="border" /></div>
              ) : (
                <>
                  {alerts.length === 0 ? (
                    <p className="text-muted">No active alerts</p>
                  ) : (
                    <div>
                      {alerts.map(alert => (
                        <Alert 
                          key={alert.id} 
                          variant={
                            alert.priority === 'high' ? 'danger' : 
                            alert.priority === 'medium' ? 'warning' : 'info'
                          }
                          className="d-flex justify-content-between align-items-center"
                        >
                          <div>
                            <div className="mb-1">
                              <Badge bg={
                                alert.type === 'signal' ? 'primary' : 
                                alert.type === 'risk' ? 'warning' : 'secondary'
                              }>
                                {alert.type.toUpperCase()}
                              </Badge>
                              <small className="ms-2 text-muted">
                                {new Date(alert.timestamp).toLocaleTimeString()}
                              </small>
                            </div>
                            {alert.message}
                          </div>
                          <Button 
                            variant="link" 
                            size="sm" 
                            className="p-0 ms-2"
                            onClick={() => handleDismissAlert(alert.id)}
                          >
                            &times;
                          </Button>
                        </Alert>
                      ))}
                    </div>
                  )}
                </>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row>
        <Col>
          <Card>
            <Card.Body>
              <div className="d-flex justify-content-between align-items-center mb-3">
                <Card.Title>Watchlist</Card.Title>
                <InputGroup style={{ width: '300px' }}>
                  <Form.Control
                    placeholder="Add symbol (e.g., TSLA)"
                    value={newWatchlistSymbol}
                    onChange={(e) => setNewWatchlistSymbol(e.target.value)}
                  />
                  <Button 
                    variant="outline-primary" 
                    onClick={handleAddToWatchlist}
                  >
                    Add
                  </Button>
                </InputGroup>
              </div>
              {loading ? (
                <div className="text-center p-5"><Spinner animation="border" /></div>
              ) : (
                <div className="table-responsive">
                  <Table hover>
                    <thead>
                      <tr>
                        <th>Symbol</th>
                        <th>Name</th>
                        <th>Price</th>
                        <th>Change</th>
                        <th>Volume</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {watchlist.map(item => (
                        <tr key={item.symbol}>
                          <td><strong>{item.symbol}</strong></td>
                          <td>{item.name}</td>
                          <td>${parseFloat(item.price).toFixed(2)}</td>
                          <td className={parseFloat(item.change) >= 0 ? 'text-success' : 'text-danger'}>
                            {parseFloat(item.change) >= 0 ? '+' : ''}{parseFloat(item.change).toFixed(2)} ({parseFloat(item.changePercent).toFixed(2)}%)
                          </td>
                          <td>{parseInt(item.volume).toLocaleString()}</td>
                          <td>
                            <Button 
                              variant="outline-danger" 
                              size="sm"
                              onClick={() => handleRemoveFromWatchlist(item.symbol)}
                            >
                              Remove
                            </Button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </Table>
                </div>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default Dashboard;
