import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';

const Navbar = (props) => {
  return (
    <nav className="navbar navbar-expand-lg">
  <div className="container-fluid">
    <Link className="navbar-brand" to="/"><img src="./logot.png" height="0px" width="10px" alt="logo"/></Link>
    <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
      <span className="navbar-toggler-icon"></span>
    </button>
    <div className="collapse navbar-collapse" id="navbarSupportedContent">
      <ul className="navbar-nav me-auto mb-2 mb-lg-0" style={{marginLeft:300, display:'flex'}}>
        <li className="nav-item" style={{marginLeft:60,marginRight:10}}>
          <Link className="nav-link active" aria-current="page" to="/">Home</Link>
        </li>
        <li className="nav-item" style={{marginLeft:0, marginRight:40}}>
          <Link className="nav-link active" to="/extract">Extract Watermark</Link>
        </li>
        
        <li className="nav-item" style={{marginLeft:30, marginRight:30}}>
          <Link className="nav-link" to="/model" >Train Model</Link>
        </li>
        <li className="nav-item" style={{marginLeft:10, marginRight:30}}>
          <Link className="nav-link" to="/work" >Working</Link>
        </li>
        <li className="nav-item" style={{marginLeft:200}}>
          <Link className="nav-link" to="/login" >Login</Link>
        </li>
      </ul>
      
    </div>
    
  </div>
</nav>
  )
}

export default Navbar
