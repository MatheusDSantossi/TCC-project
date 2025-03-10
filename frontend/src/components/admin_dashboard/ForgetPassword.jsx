import { React, useState } from "react";
import { Box } from "@mui/material";
import CustomTextField from "../forms/TextField";
import CustomPasswordField from "../forms/PasswordField";
import CustomButton from "../forms/Button";
import { Link } from "react-router-dom";
import { useForm } from "react-hook-form";
import AxiosInstance from "../AxiosInstance";
import { useNavigate } from "react-router-dom";
import CustomMessage from "./CustomMessage";

import ReactiveButton from "reactive-button";

import logo from "../../assets/logo.svg";

const ForgetPassword = () => {
  const navigate = useNavigate();
  const { handleSubmit, control } = useForm();
  const [buttonState, setButtonState] = useState("idle");

  const [ShowMessage, setShowMessage] = useState(false);

  const submission = (data) => {
    setButtonState("loading");

    AxiosInstance.post(`api/password_reset/`, {
      email: data.email,
    }).then((response) => {
      setShowMessage(true);
      setTimeout(() => {
        setButtonState("success");
      }, 2000);
    }).catch((error) => {
      setShowMessage(true);
      console.log(error);
      setTimeout(() => {
        setButtonState("error");
      }, 2000);
    });
  };
  return (
    <div className="flex items-center justify-center min-h-screen p-4">
      {ShowMessage ? (
        <CustomMessage
          text={
            "Se esse e-mail existe em nosso banco de dados você recebu um e-mail com as instruções para redefinir sua senha"
          }
          color={"#69C9AB"}
        />
      ) : null}
      <form
        className="w-full max-w-lg p-8 rounded-lg shadow-xl bg-primary"
        onSubmit={handleSubmit(submission)}
      >
        <Box className={"whiteBox"}>
          <div className="flex items-center justify-center mb-5">
            <img className="w-20 mr-4" src={logo} alt="" />
            <h2 className="mb-6 text-2xl font-semibold text-center left-40">
              Redefinir Senha
            </h2>
          </div>

          <label className="block mb-3 text-lg font-semibold text-left text-white">
            Digite o email cadastrado
          </label>
          <Box className={"itemBox"}>
            <CustomTextField label={"Email"} name={"email"} control={control} />
          </Box>

          <ReactiveButton
            className="font-bold text-primary hover:bg-yellow-300"
            buttonState={buttonState}
            color="yellow"
            type={"submit"}
            loadingText="Carregando..."
            successText={"Sucesso!"}
            idleText={"Solicitar redefinição de senha"}
            animation="true"
            rounded="true"
            width={250}
            height={40}
            size="large"
          />

          {/* <div className='flex items-center p-3 space-x-32'>
            <CustomButton label={"Request password reset"} type={"submit"} />
          </div> */}

          <Box className={"itemBox"} sx={{ flexDirection: "column" }}></Box>
        </Box>
      </form>
    </div>
  );
};

export default ForgetPassword;
