import React, { useState } from 'react';
import './Login.css';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = (event) => {
    event.preventDefault();
    // Handle form submission here
    console.log('Email:', email);
    console.log('Password:', password);
  };

  return (
    /* From Uiverse.io by AnthonyPreite */ 

  <div id="form-ui">
  <form onSubmit={handleSubmit} method="post" id="form">
    <div id="form-body">
      <div id="welcome-lines">
        <div id="welcome-line-1">Sign In</div>
        <div id="welcome-line-2">Welcome Back</div>
      </div>
      <div id="input-area">
        <div className="form-inp">
          <input placeholder="Email" type="text"/>
        </div>
        <div className="form-inp">
          <input placeholder="Password" type="password"/>
        </div>
      </div>
      <div id="submit-button-cvr">
        <button id="submit-button" type="submit">Login</button>
      </div>
      <div id="forgot-pass">
        <a href="#">Forgot password?</a>
      </div>
      <div id="bar"></div>
    </div>
  </form>
  </div>

);
};

export default Login;
