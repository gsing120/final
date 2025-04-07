import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Form, Button, Alert, Spinner } from 'react-bootstrap';

const LoginPanel = () => {
  const [activeTab, setActiveTab] = useState('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [name, setName] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [forgotPasswordEmail, setForgotPasswordEmail] = useState('');
  const [showForgotPassword, setShowForgotPassword] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [apiSecret, setApiSecret] = useState('');
  const [paperTrading, setPaperTrading] = useState(true);

  useEffect(() => {
    // Clear any errors when switching tabs
    setError('');
    setSuccess('');
  }, [activeTab]);

  const handleLogin = (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    // Validate inputs
    if (!email || !password) {
      setError('Please enter both email and password');
      setLoading(false);
      return;
    }
    
    // In a real implementation, this would call an API
    // Simulating API call with setTimeout
    setTimeout(() => {
      // For demo purposes, accept any login
      setLoading(false);
      setSuccess('Login successful! Redirecting to dashboard...');
      
      // Simulate redirect after successful login
      setTimeout(() => {
        window.location.href = '/dashboard';
      }, 1500);
    }, 1000);
  };

  const handleRegister = (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    // Validate inputs
    if (!name || !email || !password || !confirmPassword) {
      setError('Please fill in all fields');
      setLoading(false);
      return;
    }
    
    if (password !== confirmPassword) {
      setError('Passwords do not match');
      setLoading(false);
      return;
    }
    
    if (password.length < 8) {
      setError('Password must be at least 8 characters long');
      setLoading(false);
      return;
    }
    
    // In a real implementation, this would call an API
    // Simulating API call with setTimeout
    setTimeout(() => {
      setLoading(false);
      setSuccess('Registration successful! You can now log in.');
      setActiveTab('login');
      
      // Clear registration form
      setName('');
      setEmail('');
      setPassword('');
      setConfirmPassword('');
    }, 1000);
  };

  const handleForgotPassword = (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    // Validate input
    if (!forgotPasswordEmail) {
      setError('Please enter your email address');
      setLoading(false);
      return;
    }
    
    // In a real implementation, this would call an API
    // Simulating API call with setTimeout
    setTimeout(() => {
      setLoading(false);
      setSuccess('Password reset instructions have been sent to your email.');
      setShowForgotPassword(false);
      setForgotPasswordEmail('');
    }, 1000);
  };

  const handleApiSetup = (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    // Validate inputs
    if (!apiKey || !apiSecret) {
      setError('Please enter both API Key and API Secret');
      setLoading(false);
      return;
    }
    
    // In a real implementation, this would call an API to validate and store the credentials
    // Simulating API call with setTimeout
    setTimeout(() => {
      setLoading(false);
      setSuccess('API credentials saved successfully!');
      
      // Clear form
      setApiKey('');
      setApiSecret('');
    }, 1000);
  };

  const renderLoginForm = () => (
    <Form onSubmit={handleLogin}>
      <Form.Group className="mb-3">
        <Form.Label>Email address</Form.Label>
        <Form.Control
          type="email"
          placeholder="Enter email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
      </Form.Group>

      <Form.Group className="mb-3">
        <Form.Label>Password</Form.Label>
        <Form.Control
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
      </Form.Group>

      <Form.Group className="mb-3">
        <Form.Check
          type="checkbox"
          label="Remember me"
          checked={rememberMe}
          onChange={(e) => setRememberMe(e.target.checked)}
        />
      </Form.Group>

      <div className="d-grid gap-2">
        <Button variant="primary" type="submit" disabled={loading}>
          {loading ? <Spinner animation="border" size="sm" /> : 'Login'}
        </Button>
      </div>

      <div className="text-center mt-3">
        <Button
          variant="link"
          onClick={() => setShowForgotPassword(true)}
          className="p-0"
        >
          Forgot password?
        </Button>
      </div>
    </Form>
  );

  const renderRegisterForm = () => (
    <Form onSubmit={handleRegister}>
      <Form.Group className="mb-3">
        <Form.Label>Full Name</Form.Label>
        <Form.Control
          type="text"
          placeholder="Enter your name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
        />
      </Form.Group>

      <Form.Group className="mb-3">
        <Form.Label>Email address</Form.Label>
        <Form.Control
          type="email"
          placeholder="Enter email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <Form.Text className="text-muted">
          We'll never share your email with anyone else.
        </Form.Text>
      </Form.Group>

      <Form.Group className="mb-3">
        <Form.Label>Password</Form.Label>
        <Form.Control
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <Form.Text className="text-muted">
          Password must be at least 8 characters long.
        </Form.Text>
      </Form.Group>

      <Form.Group className="mb-3">
        <Form.Label>Confirm Password</Form.Label>
        <Form.Control
          type="password"
          placeholder="Confirm password"
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          required
        />
      </Form.Group>

      <div className="d-grid gap-2">
        <Button variant="primary" type="submit" disabled={loading}>
          {loading ? <Spinner animation="border" size="sm" /> : 'Register'}
        </Button>
      </div>
    </Form>
  );

  const renderApiSetupForm = () => (
    <Form onSubmit={handleApiSetup}>
      <div className="mb-3">
        <p>
          Connect your Moomoo trading account to enable live trading. Your credentials are encrypted and stored securely.
        </p>
      </div>

      <Form.Group className="mb-3">
        <Form.Label>API Key</Form.Label>
        <Form.Control
          type="text"
          placeholder="Enter your Moomoo API Key"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          required
        />
      </Form.Group>

      <Form.Group className="mb-3">
        <Form.Label>API Secret</Form.Label>
        <Form.Control
          type="password"
          placeholder="Enter your Moomoo API Secret"
          value={apiSecret}
          onChange={(e) => setApiSecret(e.target.value)}
          required
        />
      </Form.Group>

      <Form.Group className="mb-3">
        <Form.Check
          type="switch"
          id="paper-trading-switch"
          label="Enable Paper Trading Mode"
          checked={paperTrading}
          onChange={(e) => setPaperTrading(e.target.checked)}
        />
        <Form.Text className="text-muted">
          Paper trading allows you to test strategies without risking real money.
        </Form.Text>
      </Form.Group>

      <div className="d-grid gap-2">
        <Button variant="primary" type="submit" disabled={loading}>
          {loading ? <Spinner animation="border" size="sm" /> : 'Save API Settings'}
        </Button>
      </div>
    </Form>
  );

  const renderForgotPasswordForm = () => (
    <Form onSubmit={handleForgotPassword}>
      <div className="mb-3">
        <p>
          Enter your email address below and we'll send you instructions to reset your password.
        </p>
      </div>

      <Form.Group className="mb-3">
        <Form.Label>Email address</Form.Label>
        <Form.Control
          type="email"
          placeholder="Enter email"
          value={forgotPasswordEmail}
          onChange={(e) => setForgotPasswordEmail(e.target.value)}
          required
        />
      </Form.Group>

      <div className="d-flex justify-content-between">
        <Button
          variant="outline-secondary"
          onClick={() => setShowForgotPassword(false)}
        >
          Cancel
        </Button>
        <Button variant="primary" type="submit" disabled={loading}>
          {loading ? <Spinner animation="border" size="sm" /> : 'Reset Password'}
        </Button>
      </div>
    </Form>
  );

  return (
    <Container className="login-panel">
      <Row className="justify-content-center">
        <Col md={8} lg={6}>
          <Card className="shadow-sm">
            <Card.Body>
              <div className="text-center mb-4">
                <h2>Gemma Advanced Trading</h2>
                <p className="text-muted">AI-Powered Trading Platform</p>
              </div>

              {error && <Alert variant="danger">{error}</Alert>}
              {success && <Alert variant="success">{success}</Alert>}

              {showForgotPassword ? (
                renderForgotPasswordForm()
              ) : (
                <>
                  <div className="d-flex mb-4">
                    <Button
                      variant={activeTab === 'login' ? 'primary' : 'outline-secondary'}
                      className="flex-grow-1 me-2"
                      onClick={() => setActiveTab('login')}
                    >
                      Login
                    </Button>
                    <Button
                      variant={activeTab === 'register' ? 'primary' : 'outline-secondary'}
                      className="flex-grow-1 me-2"
                      onClick={() => setActiveTab('register')}
                    >
                      Register
                    </Button>
                    <Button
                      variant={activeTab === 'api' ? 'primary' : 'outline-secondary'}
                      className="flex-grow-1"
                      onClick={() => setActiveTab('api')}
                    >
                      API Setup
                    </Button>
                  </div>

                  {activeTab === 'login' && renderLoginForm()}
                  {activeTab === 'register' && renderRegisterForm()}
                  {activeTab === 'api' && renderApiSetupForm()}
                </>
              )}
            </Card.Body>
          </Card>

          <div className="text-center mt-4">
            <p className="text-muted">
              &copy; {new Date().getFullYear()} Gemma Advanced Trading. All rights reserved.
            </p>
          </div>
        </Col>
      </Row>
    </Container>
  );
};

export default LoginPanel;
