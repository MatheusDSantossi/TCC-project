import { useEffect, useState } from "react";
import logo from "../assets/logo.svg";
import AxiosInstance from "./AxiosInstance";

import ReactiveButton from "reactive-button";
import InfoDialogScreen from "./InfoDialogScreen";

function ServiceRegistration() {
  const [serviceTitle, setServiceTitle] = useState("");
  const [serviceDescription, setServiceDescription] = useState("");
  const [resetKey, setResetKey] = useState(0);

  const [buttonState, setButtonState] = useState("idle");

  const registerProduct = (e) => {
    e.preventDefault();

    setButtonState("loading");

    AxiosInstance.post(`register-product/post/`, {
      serviceTitle: serviceTitle.trim(),
      serviceDescription: serviceDescription.trim(),
    })
      .then((response) => {
        if (!response) {
          throw new Error("Failed to register the product");
        }

        setTimeout(() => {
          setButtonState("success");
        }, 2000);

        console.log("Response: ", response);
        setServiceTitle("");
        setServiceDescription("");
        setResetKey((prev) => prev + 1);
      })
      .catch((error) => {
        setButtonState("error");
        console.error("Error registering the product: ", error);
      });
  };

  return (
    <div className="flex items-center justify-center min-h-screen p-4 bg-tertiary">
      <div className="w-full max-w-lg p-8 rounded-lg shadow-xl bg-primary">
        <div className="flex items-center justify-center mb-5">
          <img className="w-20 mr-4" src={logo} alt="" />
          <h2 className="mb-6 text-2xl font-semibold text-center left-40">
            Cadastro de Serviços ou Informações
          </h2>
        </div>
        <form onSubmit={registerProduct}>
          {/* # divide the product name into three parts (name, brand, quantity) */}
          {/* Service Title */}
          <div className="mb-5">
            <div className="flex items-center">
              <label
                htmlFor="productName"
                className="block text-lg font-semibold text-white"
              >
                Título do Serviço
              </label>
              {/* <span className="p-1 rounded-full text-secondary">
                <BsInfoCircleFill />
              </span> */}
              <InfoDialogScreen
                infoText={
                  <h2>
                    Insira o título do serviço ou informação que deseja registrar.  
                    Exemplo: <strong>Nome do Gerente</strong>.  
                    No campo abaixo, descreva exatamente o que esse título representa.
                  </h2>
                }
              />
            </div>
            <input
              type="text"
              id="serviceTitle"
              name="serviceTitle"
              placeholder="Digite o nome do produto"
              required
              className="w-full p-2 mt-1 rounded-lg bg-input focus:outline-none focus:ring-2 focus:ring-yellow-400"
              value={serviceTitle}
              list="productsSuggestions"
              onChange={(e) => setServiceTitle(e.target.value)}
            />
          </div>

          {/* Service Description */}
          <div className="mb-5">
            <div className="flex items-center">
              <label
                htmlFor="productBrand"
                className="block text-lg font-semibold text-white"
              >
                Descrição do Serviço
              </label>
              <InfoDialogScreen
                infoText={
                  <h2>
                    Forneça a descrição detalhada do serviço ou informação.  
                    Exemplo: <strong>O nome do gerente é Sicrano</strong>.  
                    Seja claro e objetivo para facilitar o aprendizado do chatbot.
                  </h2>
                }
              />
            </div>
            <textarea
              id="serviceDescription"
              name="serviceDescription"
              required
              className="w-full p-4 mt-1 rounded-lg bg-input focus:outline-none focus:ring-2 focus:ring-yellow-400"
              value={serviceDescription}
              onChange={(e) => setServiceDescription(e.target.value)}
              placeholder="NESTLE"
              rows={4}
              cols={50}
            />
          </div>

          <ReactiveButton
            className="text-primary hover:bg-yellow-300"
            buttonState={buttonState}
            color="yellow"
            type={"submit"}
            loadingText="Carregando..."
            successText={"Cadastrado com sucesso"}
            idleText={"Cadastrar"}
            animation="true"
            rounded="true"
            width={120}
            height={40}
            size="large"
            // onClick={registerProduct}
          />
        </form>
      </div>
    </div>
  );
}

export default ServiceRegistration;
