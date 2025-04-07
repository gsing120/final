import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Button, Table, Badge, Spinner, Alert, Tabs, Tab, Form } from 'react-bootstrap';
import { Line, Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend } from 'chart.js';

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend);

const TradeJournal = () => {
  const [loading, setLoading] = useState(true);
  const [trades, setTrades] = useState([]);
  const [filteredTrades, setFilteredTrades] = useState([]);
  const [statistics, setStatistics] = useState(null);
  const [performanceData, setPerformanceData] = useState(null);
  const [activeTab, setActiveTab] = useState('all');
  const [dateRange, setDateRange] = useState('1m');
  const [filters, setFilters] = useState({
    symbol: '',
    strategy: '',
    tradeType: 'all',
    result: 'all',
    startDate: '',
    endDate: ''
  });
  const [error, setError] = useState(null);
  const [selectedTrade, setSelectedTrade] = useState(null);
  const [tradeNote, setTradeNote] = useState('');
  const [showNoteEditor, setShowNoteEditor] = useState(false);

  useEffect(() => {
    fetchTradeData();
  }, [dateRange]);

  useEffect(() => {
    if (trades.length > 0) {
      applyFilters();
    }
  }, [trades, filters, activeTab]);

  const fetchTradeData = async () => {
    setLoading(true);
    setError(null);

    try {
      // In a real implementation, this would be an API call to the backend
      // Simulating API call with setTimeout
      setTimeout(() => {
        // Generate mock trade data
        const mockTrades = generateMockTrades(100);
        setTrades(mockTrades);
        
        // Calculate statistics
        calculateStatistics(mockTrades);
        
        // Generate performance data
        generatePerformanceData(mockTrades);
        
        setLoading(false);
      }, 1000);
    } catch (error) {
      setError('Failed to fetch trade data. Please try again later.');
      setLoading(false);
    }
  };

  const generateMockTrades = (count) => {
    const strategies = ['Momentum Breakout', 'Volatility Mean Reversion', 'Trend Following ETF', 'News-Based Strategy', 'Gap and Go'];
    const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT', 'DIS', 'NFLX'];
    const tradeTypes = ['long', 'short'];
    
    const now = new Date();
    const threeMonthsAgo = new Date(now);
    threeMonthsAgo.setMonth(now.getMonth() - 3);
    
    return Array(count).fill().map((_, i) => {
      const entryDate = new Date(threeMonthsAgo.getTime() + Math.random() * (now.getTime() - threeMonthsAgo.getTime()));
      const holdingPeriod = Math.floor(Math.random() * 10) + 1; // 1-10 days
      const exitDate = new Date(entryDate);
      exitDate.setDate(exitDate.getDate() + holdingPeriod);
      
      const symbol = symbols[Math.floor(Math.random() * symbols.length)];
      const strategy = strategies[Math.floor(Math.random() * strategies.length)];
      const tradeType = tradeTypes[Math.floor(Math.random() * tradeTypes.length)];
      const entryPrice = parseFloat((50 + Math.random() * 200).toFixed(2));
      const exitPrice = parseFloat((entryPrice * (1 + (Math.random() * 0.2 - 0.1))).toFixed(2));
      const quantity = Math.floor(Math.random() * 100) + 1;
      
      const pnl = tradeType === 'long' 
        ? (exitPrice - entryPrice) * quantity 
        : (entryPrice - exitPrice) * quantity;
      
      const pnlPercent = tradeType === 'long'
        ? ((exitPrice / entryPrice) - 1) * 100
        : ((entryPrice / exitPrice) - 1) * 100;
      
      return {
        id: i + 1,
        symbol,
        strategy,
        tradeType,
        entryDate: entryDate.toISOString(),
        exitDate: exitDate.toISOString(),
        entryPrice,
        exitPrice,
        quantity,
        pnl,
        pnlPercent,
        fees: parseFloat((quantity * 0.005).toFixed(2)),
        holdingPeriod,
        result: pnl > 0 ? 'win' : 'loss',
        notes: Math.random() > 0.7 ? generateRandomNote() : '',
        tags: generateRandomTags(),
        screenshots: Math.random() > 0.8 ? [
          { id: 1, url: 'https://example.com/screenshot1.png', description: 'Entry point' },
          { id: 2, url: 'https://example.com/screenshot2.png', description: 'Exit point' }
        ] : [],
        lessons: Math.random() > 0.8 ? generateRandomLessons() : []
      };
    });
  };

  const generateRandomNote = () => {
    const notes = [
      "Followed the strategy rules perfectly. Good execution.",
      "Entered too early, should have waited for confirmation.",
      "Exited too soon, left money on the table.",
      "Increased position size due to high conviction setup.",
      "Reduced position size due to market volatility.",
      "Stopped out by market noise, setup was still valid.",
      "News event caused unexpected movement.",
      "Earnings announcement created volatility.",
      "Followed the plan exactly, good discipline.",
      "Broke my rules, need to stick to the plan next time."
    ];
    
    return notes[Math.floor(Math.random() * notes.length)];
  };

  const generateRandomTags = () => {
    const allTags = ['Breakout', 'Reversal', 'Trend', 'Gap', 'Earnings', 'News', 'Technical', 'Fundamental', 'Swing', 'Daytrade', 'Overextended', 'Support', 'Resistance'];
    const numTags = Math.floor(Math.random() * 3) + 1; // 1-3 tags
    const tags = [];
    
    for (let i = 0; i < numTags; i++) {
      const randomTag = allTags[Math.floor(Math.random() * allTags.length)];
      if (!tags.includes(randomTag)) {
        tags.push(randomTag);
      }
    }
    
    return tags;
  };

  const generateRandomLessons = () => {
    const lessons = [
      "Wait for confirmation before entering a trade",
      "Stick to the trading plan",
      "Don't overtrade during volatile markets",
      "Cut losses quickly",
      "Let winners run",
      "Don't trade around major news events",
      "Reduce position size when uncertain",
      "Focus on high-probability setups",
      "Be patient for the right setup",
      "Don't chase momentum"
    ];
    
    const numLessons = Math.floor(Math.random() * 2) + 1; // 1-2 lessons
    const selectedLessons = [];
    
    for (let i = 0; i < numLessons; i++) {
      const randomLesson = lessons[Math.floor(Math.random() * lessons.length)];
      if (!selectedLessons.includes(randomLesson)) {
        selectedLessons.push(randomLesson);
      }
    }
    
    return selectedLessons;
  };

  const calculateStatistics = (tradeData) => {
    if (!tradeData || tradeData.length === 0) {
      setStatistics(null);
      return;
    }
    
    const totalTrades = tradeData.length;
    const winningTrades = tradeData.filter(trade => trade.pnl > 0);
    const losingTrades = tradeData.filter(trade => trade.pnl <= 0);
    
    const winCount = winningTrades.length;
    const lossCount = losingTrades.length;
    
    const winRate = (winCount / totalTrades) * 100;
    
    const grossProfit = winningTrades.reduce((sum, trade) => sum + trade.pnl, 0);
    const grossLoss = Math.abs(losingTrades.reduce((sum, trade) => sum + trade.pnl, 0));
    
    const profitFactor = grossLoss === 0 ? Infinity : grossProfit / grossLoss;
    
    const avgWin = winCount > 0 ? grossProfit / winCount : 0;
    const avgLoss = lossCount > 0 ? grossLoss / lossCount : 0;
    
    const expectancy = (winRate / 100 * avgWin) - ((100 - winRate) / 100 * avgLoss);
    
    const netProfit = grossProfit - grossLoss;
    
    const avgHoldingPeriod = tradeData.reduce((sum, trade) => sum + trade.holdingPeriod, 0) / totalTrades;
    
    const stats = {
      totalTrades,
      winCount,
      lossCount,
      winRate: winRate.toFixed(2),
      profitFactor: profitFactor.toFixed(2),
      avgWin: avgWin.toFixed(2),
      avgLoss: avgLoss.toFixed(2),
      expectancy: expectancy.toFixed(2),
      netProfit: netProfit.toFixed(2),
      avgHoldingPeriod: avgHoldingPeriod.toFixed(1),
      largestWin: Math.max(...winningTrades.map(trade => trade.pnl), 0).toFixed(2),
      largestLoss: Math.min(...losingTrades.map(trade => trade.pnl), 0).toFixed(2)
    };
    
    setStatistics(stats);
  };

  const generatePerformanceData = (tradeData) => {
    if (!tradeData || tradeData.length === 0) {
      setPerformanceData(null);
      return;
    }
    
    // Sort trades by date
    const sortedTrades = [...tradeData].sort((a, b) => new Date(a.entryDate) - new Date(b.entryDate));
    
    // Generate equity curve
    let equity = 10000; // Starting equity
    const equityCurve = [{ date: new Date(sortedTrades[0].entryDate).toISOString().split('T')[0], equity }];
    
    sortedTrades.forEach(trade => {
      equity += trade.pnl;
      equityCurve.push({
        date: new Date(trade.exitDate).toISOString().split('T')[0],
        equity
      });
    });
    
    // Generate monthly returns
    const monthlyReturns = {};
    
    sortedTrades.forEach(trade => {
      const month = new Date(trade.exitDate).toISOString().slice(0, 7); // YYYY-MM format
      if (!monthlyReturns[month]) {
        monthlyReturns[month] = 0;
      }
      monthlyReturns[month] += trade.pnl;
    });
    
    const monthlyReturnData = Object.entries(monthlyReturns).map(([month, pnl]) => ({
      month,
      pnl
    }));
    
    // Generate win/loss streak data
    let currentStreak = 0;
    let maxWinStreak = 0;
    let maxLossStreak = 0;
    let streakType = null;
    
    sortedTrades.forEach(trade => {
      const isWin = trade.pnl > 0;
      
      if (streakType === null) {
        streakType = isWin;
        currentStreak = 1;
      } else if (streakType === isWin) {
        currentStreak++;
      } else {
        streakType = isWin;
        currentStreak = 1;
      }
      
      if (isWin) {
        maxWinStreak = Math.max(maxWinStreak, currentStreak);
      } else {
        maxLossStreak = Math.max(maxLossStreak, currentStreak);
      }
    });
    
    // Generate performance by symbol
    const symbolPerformance = {};
    
    sortedTrades.forEach(trade => {
      if (!symbolPerformance[trade.symbol]) {
        symbolPerformance[trade.symbol] = {
          totalTrades: 0,
          winCount: 0,
          lossCount: 0,
          netPnl: 0
        };
      }
      
      symbolPerformance[trade.symbol].totalTrades++;
      if (trade.pnl > 0) {
        symbolPerformance[trade.symbol].winCount++;
      } else {
        symbolPerformance[trade.symbol].lossCount++;
      }
      symbolPerformance[trade.symbol].netPnl += trade.pnl;
    });
    
    const symbolPerformanceData = Object.entries(symbolPerformance).map(([symbol, data]) => ({
      symbol,
      totalTrades: data.totalTrades,
      winRate: ((data.winCount / data.totalTrades) * 100).toFixed(2),
      netPnl: data.netPnl.toFixed(2)
    }));
    
    // Generate performance by strategy
    const strategyPerformance = {};
    
    sortedTrades.forEach(trade => {
      if (!strategyPerformance[trade.strategy]) {
        strategyPerformance[trade.strategy] = {
          totalTrades: 0,
          winCount: 0,
          lossCount: 0,
          netPnl: 0
        };
      }
      
      strategyPerformance[trade.strategy].totalTrades++;
      if (trade.pnl > 0) {
        strategyPerformance[trade.strategy].winCount++;
      } else {
        strategyPerformance[trade.strategy].lossCount++;
      }
      strategyPerformance[trade.strategy].netPnl += trade.pnl;
    });
    
    const strategyPerformanceData = Object.entries(strategyPerformance).map(([strategy, data]) => ({
      strategy,
      totalTrades: data.totalTrades,
      winRate: ((data.winCount / data.totalTrades) * 100).toFixed(2),
      netPnl: data.netPnl.toFixed(2)
    }));
    
    setPerformanceData({
      equityCurve,
      monthlyReturns: monthlyReturnData,
      maxWinStreak,
      maxLossStreak,
      symbolPerformance: symbolPerformanceData,
      strategyPerformance: strategyPerformanceData
    });
  };

  const applyFilters = () => {
    let filtered = [...trades];
    
    // Apply tab filter
    if (activeTab === 'wins') {
      filtered = filtered.filter(trade => trade.pnl > 0);
    } else if (activeTab === 'losses') {
      filtered = filtered.filter(trade => trade.pnl <= 0);
    }
    
    // Apply symbol filter
    if (filters.symbol) {
      filtered = filtered.filter(trade => 
        trade.symbol.toLowerCase().includes(filters.symbol.toLowerCase())
      );
    }
    
    // Apply strategy filter
    if (filters.strategy) {
      filtered = filtered.filter(trade => 
        trade.strategy.toLowerCase().includes(filters.strategy.toLowerCase())
      );
    }
    
    // Apply trade type filter
    if (filters.tradeType !== 'all') {
      filtered = filtered.filter(trade => trade.tradeType === filters.tradeType);
    }
    
    // Apply result filter
    if (filters.result !== 'all') {
      filtered = filtered.filter(trade => trade.result === filters.result);
    }
    
    // Apply date range filters
    if (filters.startDate) {
      const startDate = new Date(filters.startDate);
      filtered = filtered.filter(trade => new Date(trade.entryDate) >= startDate);
    }
    
    if (filters.endDate) {
      const endDate = new Date(filters.endDate);
      filtered = filtered.filter(trade => new Date(trade.entryDate) <= endDate);
    }
    
    setFilteredTrades(filtered);
    calculateStatistics(filtered);
    generatePerformanceData(filtered);
  };

  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    setFilters({
      ...filters,
      [name]: value
    });
  };

  const handleResetFilters = () => {
    setFilters({
      symbol: '',
      strategy: '',
      tradeType: 'all',
      result: 'all',
      startDate: '',
      endDate: ''
    });
  };

  const handleDateRangeChange = (range) => {
    setDateRange(range);
    
    const now = new Date();
    let startDate = '';
    
    switch (range) {
      case '1w':
        const oneWeekAgo = new Date(now);
        oneWeekAgo.setDate(now.getDate() - 7);
        startDate = oneWeekAgo.toISOString().split('T')[0];
        break;
      case '1m':
        const oneMonthAgo = new Date(now);
        oneMonthAgo.setMonth(now.getMonth() - 1);
        startDate = oneMonthAgo.toISOString().split('T')[0];
        break;
      case '3m':
        const threeMonthsAgo = new Date(now);
        threeMonthsAgo.setMonth(now.getMonth() - 3);
        startDate = threeMonthsAgo.toISOString().split('T')[0];
        break;
      case '6m':
        const sixMonthsAgo = new Date(now);
        sixMonthsAgo.setMonth(now.getMonth() - 6);
        startDate = sixMonthsAgo.toISOString().split('T')[0];
        break;
      case '1y':
        const oneYearAgo = new Date(now);
        oneYearAgo.setFullYear(now.getFullYear() - 1);
        startDate = oneYearAgo.toISOString().split('T')[0];
        break;
      case 'all':
        startDate = '';
        break;
      default:
        startDate = '';
    }
    
    setFilters({
      ...filters,
      startDate,
      endDate: now.toISOString().split('T')[0]
    });
  };

  const handleViewTrade = (trade) => {
    setSelectedTrade(trade);
    setTradeNote(trade.notes || '');
  };

  const handleCloseTradeView = () => {
    setSelectedTrade(null);
    setTradeNote('');
    setShowNoteEditor(false);
  };

  const handleSaveNote = () => {
    if (selectedTrade) {
      // In a real implementation, this would update the note in the database
      const updatedTrades = trades.map(trade => {
        if (trade.id === selectedTrade.id) {
          return {
            ...trade,
            notes: tradeNote
          };
        }
        return trade;
      });
      
      setTrades(updatedTrades);
      setSelectedTrade({
        ...selectedTrade,
        notes: tradeNote
      });
      setShowNoteEditor(false);
    }
  };

  const renderEquityCurve = () => {
    if (!performanceData || !performanceData.equityCurve) return null;

    const chartData = {
      labels: performanceData.equityCurve.map(point => point.date),
      datasets: [
        {
          label: 'Equity Curve',
          data: performanceData.equityCurve.map(point => point.equity),
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
          callbacks: {
            label: function(context) {
              return `Equity: $${context.parsed.y.toFixed(2)}`;
            }
          }
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
      }
    };

    return (
      <div style={{ height: '300px' }}>
        <Line data={chartData} options={options} />
      </div>
    );
  };

  const renderMonthlyReturns = () => {
    if (!performanceData || !performanceData.monthlyReturns) return null;

    const chartData = {
      labels: performanceData.monthlyReturns.map(item => {
        const [year, month] = item.month.split('-');
        return `${new Date(0, parseInt(month) - 1).toLocaleString('default', { month: 'short' })} ${year}`;
      }),
      datasets: [
        {
          label: 'Monthly P&L',
          data: performanceData.monthlyReturns.map(item => item.pnl),
          backgroundColor: performanceData.monthlyReturns.map(item => 
            item.pnl >= 0 ? 'rgba(75, 192, 192, 0.7)' : 'rgba(255, 99, 132, 0.7)'
          ),
          borderColor: performanceData.monthlyReturns.map(item => 
            item.pnl >= 0 ? 'rgba(75, 192, 192, 1)' : 'rgba(255, 99, 132, 1)'
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
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return `P&L: $${context.parsed.y.toFixed(2)}`;
            }
          }
        }
      },
      scales: {
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
      }
    };

    return (
      <div style={{ height: '300px' }}>
        <Bar data={chartData} options={options} />
      </div>
    );
  };

  const renderTradeDetails = () => {
    if (!selectedTrade) return null;

    return (
      <Card className="mb-4">
        <Card.Header className="d-flex justify-content-between align-items-center">
          <h5 className="mb-0">Trade Details</h5>
          <Button variant="outline-secondary" size="sm" onClick={handleCloseTradeView}>Close</Button>
        </Card.Header>
        <Card.Body>
          <Row>
            <Col md={6}>
              <Table bordered>
                <tbody>
                  <tr>
                    <td><strong>Symbol</strong></td>
                    <td>{selectedTrade.symbol}</td>
                  </tr>
                  <tr>
                    <td><strong>Strategy</strong></td>
                    <td>{selectedTrade.strategy}</td>
                  </tr>
                  <tr>
                    <td><strong>Trade Type</strong></td>
                    <td>{selectedTrade.tradeType.toUpperCase()}</td>
                  </tr>
                  <tr>
                    <td><strong>Entry Date</strong></td>
                    <td>{new Date(selectedTrade.entryDate).toLocaleString()}</td>
                  </tr>
                  <tr>
                    <td><strong>Exit Date</strong></td>
                    <td>{new Date(selectedTrade.exitDate).toLocaleString()}</td>
                  </tr>
                  <tr>
                    <td><strong>Holding Period</strong></td>
                    <td>{selectedTrade.holdingPeriod} days</td>
                  </tr>
                </tbody>
              </Table>
            </Col>
            <Col md={6}>
              <Table bordered>
                <tbody>
                  <tr>
                    <td><strong>Entry Price</strong></td>
                    <td>${selectedTrade.entryPrice.toFixed(2)}</td>
                  </tr>
                  <tr>
                    <td><strong>Exit Price</strong></td>
                    <td>${selectedTrade.exitPrice.toFixed(2)}</td>
                  </tr>
                  <tr>
                    <td><strong>Quantity</strong></td>
                    <td>{selectedTrade.quantity}</td>
                  </tr>
                  <tr>
                    <td><strong>P&L</strong></td>
                    <td className={selectedTrade.pnl >= 0 ? 'text-success' : 'text-danger'}>
                      ${selectedTrade.pnl.toFixed(2)} ({selectedTrade.pnlPercent.toFixed(2)}%)
                    </td>
                  </tr>
                  <tr>
                    <td><strong>Fees</strong></td>
                    <td>${selectedTrade.fees.toFixed(2)}</td>
                  </tr>
                  <tr>
                    <td><strong>Net P&L</strong></td>
                    <td className={(selectedTrade.pnl - selectedTrade.fees) >= 0 ? 'text-success' : 'text-danger'}>
                      ${(selectedTrade.pnl - selectedTrade.fees).toFixed(2)}
                    </td>
                  </tr>
                </tbody>
              </Table>
            </Col>
          </Row>

          <Row className="mt-3">
            <Col md={12}>
              <Card>
                <Card.Header className="d-flex justify-content-between align-items-center">
                  <h6 className="mb-0">Trade Notes</h6>
                  {!showNoteEditor && (
                    <Button 
                      variant="outline-primary" 
                      size="sm"
                      onClick={() => setShowNoteEditor(true)}
                    >
                      Edit
                    </Button>
                  )}
                </Card.Header>
                <Card.Body>
                  {showNoteEditor ? (
                    <>
                      <Form.Control
                        as="textarea"
                        rows={4}
                        value={tradeNote}
                        onChange={(e) => setTradeNote(e.target.value)}
                        placeholder="Enter notes about this trade..."
                      />
                      <div className="mt-2 text-end">
                        <Button 
                          variant="outline-secondary" 
                          size="sm"
                          className="me-2"
                          onClick={() => {
                            setTradeNote(selectedTrade.notes || '');
                            setShowNoteEditor(false);
                          }}
                        >
                          Cancel
                        </Button>
                        <Button 
                          variant="primary" 
                          size="sm"
                          onClick={handleSaveNote}
                        >
                          Save
                        </Button>
                      </div>
                    </>
                  ) : (
                    <div>
                      {selectedTrade.notes ? (
                        <p>{selectedTrade.notes}</p>
                      ) : (
                        <p className="text-muted">No notes for this trade.</p>
                      )}
                    </div>
                  )}
                </Card.Body>
              </Card>
            </Col>
          </Row>

          {selectedTrade.tags && selectedTrade.tags.length > 0 && (
            <Row className="mt-3">
              <Col md={12}>
                <h6>Tags</h6>
                <div>
                  {selectedTrade.tags.map((tag, index) => (
                    <Badge key={index} bg="secondary" className="me-2">{tag}</Badge>
                  ))}
                </div>
              </Col>
            </Row>
          )}

          {selectedTrade.lessons && selectedTrade.lessons.length > 0 && (
            <Row className="mt-3">
              <Col md={12}>
                <h6>Lessons Learned</h6>
                <ul>
                  {selectedTrade.lessons.map((lesson, index) => (
                    <li key={index}>{lesson}</li>
                  ))}
                </ul>
              </Col>
            </Row>
          )}

          {selectedTrade.screenshots && selectedTrade.screenshots.length > 0 && (
            <Row className="mt-3">
              <Col md={12}>
                <h6>Screenshots</h6>
                <div className="d-flex flex-wrap">
                  {selectedTrade.screenshots.map((screenshot, index) => (
                    <div key={index} className="me-3 mb-3">
                      <div className="border p-1">
                        <div style={{ width: '200px', height: '150px', backgroundColor: '#f0f0f0', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          <span className="text-muted">Screenshot Preview</span>
                        </div>
                        <div className="mt-1 text-center">
                          <small>{screenshot.description}</small>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </Col>
            </Row>
          )}
        </Card.Body>
      </Card>
    );
  };

  return (
    <Container fluid className="trade-journal">
      <Row className="mb-3">
        <Col>
          <div className="d-flex justify-content-between align-items-center">
            <h2>Trade Journal</h2>
            <div>
              <Button 
                variant="outline-secondary" 
                size="sm" 
                className="me-2" 
                onClick={() => handleDateRangeChange('1w')}
              >
                1W
              </Button>
              <Button 
                variant="outline-secondary" 
                size="sm" 
                className="me-2" 
                onClick={() => handleDateRangeChange('1m')}
              >
                1M
              </Button>
              <Button 
                variant="outline-secondary" 
                size="sm" 
                className="me-2" 
                onClick={() => handleDateRangeChange('3m')}
              >
                3M
              </Button>
              <Button 
                variant="outline-secondary" 
                size="sm" 
                className="me-2" 
                onClick={() => handleDateRangeChange('6m')}
              >
                6M
              </Button>
              <Button 
                variant="outline-secondary" 
                size="sm" 
                className="me-2" 
                onClick={() => handleDateRangeChange('1y')}
              >
                1Y
              </Button>
              <Button 
                variant="outline-secondary" 
                size="sm" 
                onClick={() => handleDateRangeChange('all')}
              >
                All
              </Button>
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

      {selectedTrade && renderTradeDetails()}

      <Row className="mb-4">
        <Col lg={8}>
          <Card className="h-100">
            <Card.Body>
              <Card.Title>Performance Overview</Card.Title>
              {loading ? (
                <div className="text-center p-5"><Spinner animation="border" /></div>
              ) : (
                <>
                  {renderEquityCurve()}
                  <Row className="mt-3">
                    <Col md={3} className="text-center">
                      <div className="text-muted">Total Trades</div>
                      <h4>{statistics?.totalTrades || 0}</h4>
                    </Col>
                    <Col md={3} className="text-center">
                      <div className="text-muted">Win Rate</div>
                      <h4>{statistics?.winRate || 0}%</h4>
                    </Col>
                    <Col md={3} className="text-center">
                      <div className="text-muted">Profit Factor</div>
                      <h4>{statistics?.profitFactor || 0}</h4>
                    </Col>
                    <Col md={3} className="text-center">
                      <div className="text-muted">Net Profit</div>
                      <h4 className={parseFloat(statistics?.netProfit || 0) >= 0 ? 'text-success' : 'text-danger'}>
                        ${parseFloat(statistics?.netProfit || 0).toLocaleString()}
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
              <Card.Title>Monthly Returns</Card.Title>
              {loading ? (
                <div className="text-center p-5"><Spinner animation="border" /></div>
              ) : (
                renderMonthlyReturns()
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row className="mb-4">
        <Col md={6}>
          <Card className="h-100">
            <Card.Body>
              <Card.Title>Trade Statistics</Card.Title>
              {loading ? (
                <div className="text-center p-5"><Spinner animation="border" /></div>
              ) : (
                <Table striped bordered>
                  <tbody>
                    <tr>
                      <td>Total Trades</td>
                      <td>{statistics?.totalTrades || 0}</td>
                    </tr>
                    <tr>
                      <td>Winning Trades</td>
                      <td>{statistics?.winCount || 0}</td>
                    </tr>
                    <tr>
                      <td>Losing Trades</td>
                      <td>{statistics?.lossCount || 0}</td>
                    </tr>
                    <tr>
                      <td>Win Rate</td>
                      <td>{statistics?.winRate || 0}%</td>
                    </tr>
                    <tr>
                      <td>Profit Factor</td>
                      <td>{statistics?.profitFactor || 0}</td>
                    </tr>
                    <tr>
                      <td>Average Win</td>
                      <td>${parseFloat(statistics?.avgWin || 0).toLocaleString()}</td>
                    </tr>
                    <tr>
                      <td>Average Loss</td>
                      <td>${parseFloat(statistics?.avgLoss || 0).toLocaleString()}</td>
                    </tr>
                    <tr>
                      <td>Largest Win</td>
                      <td>${parseFloat(statistics?.largestWin || 0).toLocaleString()}</td>
                    </tr>
                    <tr>
                      <td>Largest Loss</td>
                      <td>${parseFloat(statistics?.largestLoss || 0).toLocaleString()}</td>
                    </tr>
                    <tr>
                      <td>Average Holding Period</td>
                      <td>{statistics?.avgHoldingPeriod || 0} days</td>
                    </tr>
                    <tr>
                      <td>Max Win Streak</td>
                      <td>{performanceData?.maxWinStreak || 0}</td>
                    </tr>
                    <tr>
                      <td>Max Loss Streak</td>
                      <td>{performanceData?.maxLossStreak || 0}</td>
                    </tr>
                  </tbody>
                </Table>
              )}
            </Card.Body>
          </Card>
        </Col>
        <Col md={6}>
          <Card className="h-100">
            <Card.Body>
              <Tabs
                activeKey={activeTab}
                onSelect={(k) => setActiveTab(k)}
                className="mb-3"
              >
                <Tab eventKey="symbol" title="By Symbol">
                  {loading ? (
                    <div className="text-center p-5"><Spinner animation="border" /></div>
                  ) : (
                    <div className="table-responsive">
                      <Table striped bordered hover>
                        <thead>
                          <tr>
                            <th>Symbol</th>
                            <th>Trades</th>
                            <th>Win Rate</th>
                            <th>Net P&L</th>
                          </tr>
                        </thead>
                        <tbody>
                          {performanceData?.symbolPerformance?.map((item, index) => (
                            <tr key={index}>
                              <td>{item.symbol}</td>
                              <td>{item.totalTrades}</td>
                              <td>{item.winRate}%</td>
                              <td className={parseFloat(item.netPnl) >= 0 ? 'text-success' : 'text-danger'}>
                                ${parseFloat(item.netPnl).toLocaleString()}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </Table>
                    </div>
                  )}
                </Tab>
                <Tab eventKey="strategy" title="By Strategy">
                  {loading ? (
                    <div className="text-center p-5"><Spinner animation="border" /></div>
                  ) : (
                    <div className="table-responsive">
                      <Table striped bordered hover>
                        <thead>
                          <tr>
                            <th>Strategy</th>
                            <th>Trades</th>
                            <th>Win Rate</th>
                            <th>Net P&L</th>
                          </tr>
                        </thead>
                        <tbody>
                          {performanceData?.strategyPerformance?.map((item, index) => (
                            <tr key={index}>
                              <td>{item.strategy}</td>
                              <td>{item.totalTrades}</td>
                              <td>{item.winRate}%</td>
                              <td className={parseFloat(item.netPnl) >= 0 ? 'text-success' : 'text-danger'}>
                                ${parseFloat(item.netPnl).toLocaleString()}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </Table>
                    </div>
                  )}
                </Tab>
              </Tabs>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row>
        <Col>
          <Card>
            <Card.Body>
              <div className="d-flex justify-content-between align-items-center mb-3">
                <Tabs
                  activeKey={activeTab}
                  onSelect={(k) => setActiveTab(k)}
                  className="mb-0"
                >
                  <Tab eventKey="all" title="All Trades" />
                  <Tab eventKey="wins" title="Winning Trades" />
                  <Tab eventKey="losses" title="Losing Trades" />
                </Tabs>
                <div>
                  <Button 
                    variant="outline-secondary" 
                    size="sm"
                    onClick={handleResetFilters}
                    className="me-2"
                  >
                    Reset Filters
                  </Button>
                  <Button 
                    variant="outline-primary" 
                    size="sm"
                    onClick={fetchTradeData}
                    disabled={loading}
                  >
                    {loading ? <Spinner animation="border" size="sm" /> : 'Refresh'}
                  </Button>
                </div>
              </div>

              <Row className="mb-3">
                <Col md={2}>
                  <Form.Group>
                    <Form.Label>Symbol</Form.Label>
                    <Form.Control
                      type="text"
                      name="symbol"
                      value={filters.symbol}
                      onChange={handleFilterChange}
                      placeholder="Filter by symbol"
                      size="sm"
                    />
                  </Form.Group>
                </Col>
                <Col md={3}>
                  <Form.Group>
                    <Form.Label>Strategy</Form.Label>
                    <Form.Control
                      type="text"
                      name="strategy"
                      value={filters.strategy}
                      onChange={handleFilterChange}
                      placeholder="Filter by strategy"
                      size="sm"
                    />
                  </Form.Group>
                </Col>
                <Col md={2}>
                  <Form.Group>
                    <Form.Label>Trade Type</Form.Label>
                    <Form.Select
                      name="tradeType"
                      value={filters.tradeType}
                      onChange={handleFilterChange}
                      size="sm"
                    >
                      <option value="all">All</option>
                      <option value="long">Long</option>
                      <option value="short">Short</option>
                    </Form.Select>
                  </Form.Group>
                </Col>
                <Col md={2}>
                  <Form.Group>
                    <Form.Label>Result</Form.Label>
                    <Form.Select
                      name="result"
                      value={filters.result}
                      onChange={handleFilterChange}
                      size="sm"
                    >
                      <option value="all">All</option>
                      <option value="win">Win</option>
                      <option value="loss">Loss</option>
                    </Form.Select>
                  </Form.Group>
                </Col>
                <Col md={3}>
                  <Row>
                    <Col md={6}>
                      <Form.Group>
                        <Form.Label>Start Date</Form.Label>
                        <Form.Control
                          type="date"
                          name="startDate"
                          value={filters.startDate}
                          onChange={handleFilterChange}
                          size="sm"
                        />
                      </Form.Group>
                    </Col>
                    <Col md={6}>
                      <Form.Group>
                        <Form.Label>End Date</Form.Label>
                        <Form.Control
                          type="date"
                          name="endDate"
                          value={filters.endDate}
                          onChange={handleFilterChange}
                          size="sm"
                        />
                      </Form.Group>
                    </Col>
                  </Row>
                </Col>
              </Row>

              {loading ? (
                <div className="text-center p-5"><Spinner animation="border" /></div>
              ) : (
                <div className="table-responsive">
                  <Table hover>
                    <thead>
                      <tr>
                        <th>Symbol</th>
                        <th>Strategy</th>
                        <th>Type</th>
                        <th>Entry Date</th>
                        <th>Exit Date</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>Quantity</th>
                        <th>P&L</th>
                        <th>P&L %</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredTrades.length === 0 ? (
                        <tr>
                          <td colSpan="11" className="text-center">No trades found matching your filters</td>
                        </tr>
                      ) : (
                        filteredTrades.map(trade => (
                          <tr key={trade.id}>
                            <td>{trade.symbol}</td>
                            <td>{trade.strategy}</td>
                            <td>
                              <Badge bg={trade.tradeType === 'long' ? 'success' : 'danger'}>
                                {trade.tradeType.toUpperCase()}
                              </Badge>
                            </td>
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
                            <td>
                              <Button 
                                variant="outline-primary" 
                                size="sm"
                                onClick={() => handleViewTrade(trade)}
                              >
                                View
                              </Button>
                            </td>
                          </tr>
                        ))
                      )}
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

export default TradeJournal;
