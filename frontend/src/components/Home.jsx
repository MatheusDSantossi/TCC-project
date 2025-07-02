import '../App.css';
import React, { useEffect } from "react";
import logo from "../assets/logo.svg";
import { useNavigate, useParams } from "react-router-dom";
import AxiosInstance from "./AxiosInstance";

import hugeRobotHand from "../assets/huge_robot_hand.png";
import hugeRobotHand2 from "../assets/huge_robot_hand_2.png";
import { motion } from "framer-motion";

// TODO: Create a screen to the / url where I want to allow the client to type the name of their business to access it from this page.

const Home = () => {
  const navigate = useNavigate();
  const { businessSlug } = useParams();
  const url = `/chatbot/${businessSlug}`;

  const TestAPI = () => {
    AxiosInstance.get("http://127.0.0.1:8000/api/sample/")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => console.log(data))
      .catch((error) => console.error("Error fetching data:", error));
  };

  useEffect(() => {
    TestAPI;
  }, []);

  const handleClick = () => {
    navigate(url);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-40 drop-shadow-3xl">
      <div className="p-6 rounded-lg bg-primary">
        <div className="flex items-center justify-center mb-6">
          <img src={logo} alt="logo" className="h-20" />
        </div>
        <div className="absolute inset-0 flex justify-end -z-10">
          <motion.img
            src={hugeRobotHand}
            alt="Huge robot hand pointing"
            className="max-w-[50%] md:max-w-[40%] lg:max-w-[30%] xl:max-w-[25%] object-contain"
            initial={{ x: "100vw", opacity: 0 }}
            animate={{ x: "0", opacity: 1 }}
            transition={{ type: "spring", stiffness: 50, damping: 15 }}
          />
        </div>
        <div className="absolute inset-0 flex justify-start -z-10">
          <motion.img
            src={hugeRobotHand2}
            alt="Huge robot hand pointing"
            className="max-w-[50%] md:max-w-[40%] lg:max-w-[30%] xl:max-w-[25%] object-contain"
            initial={{ x: "-100vw", opacity: 0 }}
            animate={{ x: "0", opacity: 1 }}
            transition={{ type: "spring", stiffness: 50, damping: 15 }}
          />
        </div>
        <div className="text-center">
          <h1 className="mb-4 text-4xl font-bold">Precisa de Ajuda?</h1>
          {/* <p className='mb-8'>Selecione a opção que melhor lhe atende</p> */}
          <div className="flex justify-center space-x-4">
            <button
              className="px-6 py-3 font-bold text-black transition bg-yellow-400 rounded-lg cursor-pointer hover:bg-yellow-300"
              onClick={handleClick}
            >
              Fale com o Hecate
            </button>
            {/* <button className='px-6 py-3 font-bold text-black transition bg-yellow-400 rounded-lg hover:bg-yellow-300'
                        onClick={handleClick}
                        >
                            Dúvidas sobre o mercado
                        </button> */}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
