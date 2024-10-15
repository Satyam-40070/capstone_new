import React from "react";
import Navbar from "./Components/Navbar/Navbar";
import {BrowserRouter, Route, Routes} from "react-router-dom";
import Home from "./pages/Home/Home";
import Extract from "./pages/Extract/Extract";
import Model from "./pages/Model/Model";
import Work from "./pages/Working/Work";
import Login from "./pages/Auth/Login";
import './App.css';

function App() {
  return (
    <>
    
      <BrowserRouter>
      <Navbar/>
          <Routes>
            <Route exact path="/" element={<Home/>} />
                
            <Route path="/home" element={<Home/>} />
            <Route path="/extract" element={<Extract/>} />
            <Route path="/model" element={<Model/>} />
            <Route path="/work" element={<Work/>} />
            <Route path="/login" element={<Login/>} />
          </Routes>
        
      </BrowserRouter>
    </>
  );
}

export default App;
