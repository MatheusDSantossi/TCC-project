import logo from "../../assets/logo.svg";
import hugeRobotHand from "../../assets/huge_robot_hand.png";
import hugeRobotHand2 from "../../assets/huge_robot_hand_2.png";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

import { businessName } from "../../utils/BusinessInfo";

const AdminHome = () => {
  const navigate = useNavigate();

  const handleClick = (e, state) => {
    console.log("E.value: ", state);
    if (state === "Login") {
      navigate("/login");
    } else {
      navigate("/register");
    }
  };

  return (
    <div className="overflow-hidden">
      <div className="p-3 drop-shadow-3xl bg-primary">
        <div className="flex items-center justify-between">
          <img src={logo} alt="logo" className="h-10" />
          <div className="flex">
            <button
              onClick={(e) => handleClick(e, "Login")}
              className="p-1 mr-3 text-white rounded-lg text-md bg-quartiary hover:text-secondary-dark hover:bg-quartiary-dark"
            >
              Login
            </button>
            <button
              onClick={(e) => handleClick(e)}
              className="p-1 mr-3 text-white rounded-lg text-md bg-quartiary hover:text-secondary-dark hover:bg-quartiary-dark"
            >
              Cadastrar
            </button>
          </div>
        </div>
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
      <div className="flex flex-col items-center justify-center min-h-screen p-40">
        <div className="p-6 rounded-lg drop-shadow-3xl bg-primary w-[90%] max-w-[600px]">
          <div className="flex items-center justify-center mb-6">
            {/* <img src={logo} alt="logo" className="h-20" /> */}
          </div>
          <div className="text-center">
            <h1 className="m-6 text-4xl font-bold">
              Seja bem-vindo as Ferramentas Administrativas do
              <span className="text-secondary"> {businessName}💼</span>
            </h1>
            <hr
              style={{
                height: "1px",
                border: "none",
                backgroundColor: "#5D76D3",
              }}
            ></hr>
            <h2 className="mt-4 text-xl">
              O sistema administrativo do{" "}
              <span className="text-secondary">{businessName}</span> com
              diversas funcionalidades
            </h2>
            <h3 className="mt-6">
              Em qualquer dúvida entre em contato com o email abaixo:
              <br />
              <a className="text-secondary" href="mailto:admin@hecate.com.br">
                admin@hecate.com.br
              </a>
            </h3>
            {/* <p className='mb-8'>Selecione a opção que melhor lhe atende</p> */}
            {/* <div className="flex justify-center space-x-4">
              <button
                className="px-6 py-3 font-bold text-black transition bg-yellow-400 rounded-lg hover:bg-yellow-300"
                onClick={(e) => handleClick(e, "Login")}
              >
                Login
              </button>
              <button
                className="px-6 py-3 font-bold text-black transition bg-yellow-400 rounded-lg hover:bg-yellow-300"
                onClick={(e) => handleClick(e)}
              >
                Cadastrar
              </button>
            </div> */}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminHome;
