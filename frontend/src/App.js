import React from "react";
import VerticalNavbar from "./Components/Navbar/Navbar";
import {BrowserRouter, Route, Routes} from "react-router-dom";
import Home from "./pages/Home/Home";
import Extract from "./pages/Extract/Extract";
import Model from "./pages/Model/Model";
import Work from "./pages/Working/Work";
import Login from "./pages/Auth/Login";
import Xray from "./pages/Home/xray";
import Doc from "./pages/Home/doc";
import XrayE from "./pages/Extract/XrayE";
import DocE from "./pages/Extract/DocE";
import './App.css';

function App() {
  return (
    <>
    
      <BrowserRouter>
      <VerticalNavbar/>
          <Routes>
            <Route exact path="/" element={<Home/>} />
                
            <Route path="/home" element={<Home/>} />
            <Route path="/xray" element={<Xray/>} />
            <Route path="/doc" element={<Doc/>} />
            <Route path="/xrayex" element={<XrayE/>} />
            <Route path="/docex" element={<DocE/>} />
            <Route path="/extractwatermark" element={<Extract/>} />
            <Route path="/trainmodel" element={<Model/>} />
            <Route path="/working" element={<Work/>} />
            <Route path="/aboutus" element={<Login/>} />
          </Routes>
        
      </BrowserRouter>
    </>
  );
}

export default App;
