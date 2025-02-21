import { React, useState } from "react";
import { Box } from "@mui/material";
import CustomTextField from "../forms/TextField";
import CustomPasswordField from "../forms/PasswordField";
import CustomButton from "../forms/Button";
import { Link } from "react-router-dom";
import { useForm } from "react-hook-form";
import AxiosInstance from "../AxiosInstance";
import { useNavigate, useParams } from "react-router-dom";
import CustomMessage from "./CustomMessage";

import logo from "../../assets/logo.svg";
import ReactiveButton from "reactive-button";

const ForgetPasswordReset = () => {
  const navigate = useNavigate();
  const { handleSubmit, control } = useForm();
  const { token } = useParams();
  console.log(token);

  const [ShowMessage, setShowMessage] = useState(false);
  const [buttonState, setButtonState] = useState("idle");

  const submission = (data) => {
    setButtonState("loading");
    AxiosInstance.post(`api/password_reset/confirm/`, {
      password: data.password,
      token: token,
    }).then((response) => {
      setShowMessage(true);
      setTimeout(() => {
        setButtonState("success");
      }, 2000);
      setTimeout(() => {
        navigate("/login");
      }, 6000);
    });
  };
  return (
    <div className="flex items-center justify-center min-h-screen p-4">
      {ShowMessage ? (
        <CustomMessage
          text={
            "Sua senha foi redefinida com sucesso. Você será redirecionado para a página de login em um segundo."
          }
          color={"#69C9AB"}
        />
      ) : null}
      <form
        onSubmit={handleSubmit(submission)}
        className="w-full max-w-lg p-8 rounded-lg shadow-xl bg-primary"
      >
        <Box className={"whiteBox"}>
          <div className="flex items-center justify-center mb-5">
            <img className="w-20 mr-4" src={logo} alt="" />
            <h2 className="mb-6 text-2xl font-semibold text-center left-40">
              {" "}
              Redefina sua senha{" "}
            </h2>
          </div>

          <Box className={"itemBox"}>
            <CustomPasswordField
              label={"Senha"}
              name={"password"}
              control={control}
            />
          </Box>

          <Box className={"itemBox"}>
            <CustomPasswordField
              label={"Confirmar Senha"}
              name={"password2"}
              control={control}
            />
          </Box>

          <ReactiveButton
            className="font-bold text-primary hover:bg-yellow-300"
            buttonState={buttonState}
            color="yellow"
            type={"submit"}
            loadingText="Carregando..."
            successText={"Sucesso!"}
            idleText={"Redefinir senha"}
            animation="true"
            rounded="true"
            width={250}
            height={40}
            size="large"
          />

          <Box className={"itemBox"} sx={{ flexDirection: "column" }}></Box>
        </Box>
      </form>
    </div>
  );
};

export default ForgetPasswordReset;
