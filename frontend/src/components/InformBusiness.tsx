import { useEffect, useState } from "react";
import { Box } from "@mui/material";
import CustomTextField from "./forms/TextField";
import { useForm } from "react-hook-form";
import { useNavigate } from "react-router-dom";
import CustomMessage from "./admin_dashboard/CustomMessage";

import logo from "../assets/logo.svg";

import "../App.css";

import ReactiveButton from "reactive-button";
import AxiosInstance from "./AxiosInstance.js";

const InformBusiness = () => {
  const navigate = useNavigate();
  const { handleSubmit, control } = useForm();
  const [showMessage, setShowMessage] = useState(false);
  const [businessSlugs, setBusinessSlugs] = useState([]);
  const [buttonState, setButtonState] = useState("idle");

  useEffect(() => {
    AxiosInstance.get(`business/`)
      .then((response) => {
        setBusinessSlugs(response.data);
      })
      .catch((error) => console.error("Error fetching business data: ", error));
  }, []);

  const establishmentAccess = (data) => {
    setButtonState("loading");
    setShowMessage(false);

    let dataSlug = data.establishmentSlug.trim();

    const userSlug = dataSlug === undefined ? "default" : dataSlug;

    console.log(userSlug);

    const match = businessSlugs.find((b) => b.slug === userSlug);

    if (match) {
      setButtonState("success");
      navigate(`/${userSlug}`);
    } else {
      setButtonState("error");
      setShowMessage(true);
    }
  };

  const handleClose = (_, reason) => {
    if (reason === "clickaway") return;
    setShowMessage(false);
  };

  return (
    <div className="flex items-center justify-center min-h-screen p-4 drop-shadow-3xl">
      {showMessage ? (
        <CustomMessage
          open={showMessage}
          onClose={handleClose}
          severity="error"
          message={
            "Estabelecimento não encontrado. Verifique o slug e tente novamente"
          }
          duration={3000}
        />
      ) : null}
      <form
        className="w-full max-w-lg p-8 rounded-lg shadow-xl bg-primary"
        onSubmit={handleSubmit(establishmentAccess)}
      >
        <div className="w-full max-w-lg p-8">
          <div className="flex items-center justify-center mb-5">
            <img className="w-20 mr-4" src={logo} alt="" />
            <h2 className="mb-6 text-2xl font-semibold text-center left-40">
              Informe o nome do estabelecimento
            </h2>
          </div>
          <Box className={"itemBox mb-3"}>
            <CustomTextField
              label={"Nome do estabelecimento"}
              name={"establishmentSlug"}
              control={control}
              autoFocus={true}
            />
          </Box>
          <div className="flex items-center justify-center p-3 space-x-16">
            <ReactiveButton
              className="text-primary hover:bg-yellow-300"
              buttonState={buttonState}
              color="yellow"
              type={"submit"}
              loadingText="Carregando..."
              successText={"Estabelecimento encontrado"}
              idleText={"Acessar"}
              animation={true}
              rounded={true}
              width={80}
              height={"40"}
              size="large"
            />
          </div>
        </div>
      </form>
    </div>
  );
};

export default InformBusiness;
