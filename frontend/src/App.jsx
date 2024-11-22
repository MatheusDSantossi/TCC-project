import { BrowserRouter, Routes, Route } from "react-router-dom";
import { useEffect } from "react";

import Home from "./components/Home";
// import StarsCanvas from "./components/canvas/Stars";
import Chatbot from "./components/Chatbot";

import { RocketCanvas, StarsCanvas } from "./components/canvas";

function App() {
  return (
    <div>
      <BrowserRouter>
        <div className="relative z-10" >
          <Routes>
            <Route exact path="/" element={<Home />} />
            <Route exact path="/chatbot" element={<Chatbot />} />
          </Routes>
        </div>
        <div className="relative z-0">
          <StarsCanvas />
        </div>
        <div className="relative z-0">
          <RocketCanvas />
        </div>
      </BrowserRouter>

    </div>
  )
}

export default App;
