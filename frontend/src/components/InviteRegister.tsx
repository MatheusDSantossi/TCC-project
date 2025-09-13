import { useEffect } from "react";
import AxiosInstance from "./AxiosInstance";
import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useForm } from "react-hook-form";
import CustomPasswordField from "./forms/PasswordField";
import { Box } from "@mui/material";
import ReactiveButton from "reactive-button";
import logo from "../assets/logo.svg";

import { yupResolver } from "@hookform/resolvers/yup";
import * as yup from "yup";
import { routes } from "../routes/routes";

const InviteRegister = () => {
  // route
  const loginRoute = routes.form.login;
  
  const schema = yup.object({
    password: yup
      .string()
      .required("Senha é um campo obrigatório")
      .min(8, "Senha deve ter pelo menos 8 caracteres")
      .matches(/[A-Z]/, "Senha deve ter pelo menos uma letra maiúscula")
      .matches(/[a-z]/, "Senha deve ter pelo menos uma letra minuscúla")
      .matches(/[0-9]/, "Senha deve ter pelo menos um número")
      .matches(
        /[!@#$%^&*(),.?":;{}|<>+]/,
        "Senha deve ter pelo menos um caracter especial"
      ),
    password2: yup
      .string()
      .required("O campo de confirmação de senha é obrigatorio")
      .oneOf([yup.ref("password"), null], "As senhas devem ser iguais"),
  });
  
  const [email, setEmail] = useState("");
  const [business, setBusiness] = useState("");
  const [inviteValid, setInviteValid] = useState(false);
  const { token } = useParams();
  const navigate = useNavigate();
  const { control, handleSubmit } = useForm({ resolver: yupResolver(schema) });
  const [buttonState, setButtonState] = useState("idle");

  // 1) on mout, verify the invite and fetch email/business
  useEffect(() => {
    AxiosInstance.get(`invite/${token}/`)
      .then((response) => {
        setEmail(response.data.email);
        setBusiness(response.data.business);
        setInviteValid(true);
      })
      .catch(() => setInviteValid(false));
  }, [token]);

  // 2) on submit, POST password + token
  const onSubmit = (data) => {
    if (!inviteValid) {
      setButtonState("error");
      alert("Invalid or expired invite link.");
    }
    AxiosInstance.post(`register/invite/`, {
      token: token,
      password: data.password,
    })
      .then(() => {
        alert("Registration successful!");
        setButtonState("success");
        // Redirect
        navigate(loginRoute);
      })
      .catch((error) => {
        console.error("Registration failed:", error);
        alert("Registration failed, please try again.");
      });
  };

  return (
    <div className="flex items-center justify-center min-h-screen p-4">
      {inviteValid ? (
        <form
          onSubmit={handleSubmit(onSubmit)}
          className="w-full max-w-lg p-8 rounded-lg shadow-xl bg-primary"
        >
          <div className="flex items-center justify-center mb-5">
            <img className="w-20 mr-4" src={logo} alt="" />
            <h2 className="mb-6 text-xl font-semibold text-center left-40">
              Bem-vindo, {email}
              <br />
              <span className="text-quartiary">Empresa: {business}</span>
            </h2>
          </div>

          {/* Reuse your CustomPasswordField component */}
          <Box className={"itemBox  mb-3"}>
            <CustomPasswordField
              label={"Senha"}
              name={"password"}
              control={control}
              defaultValue=""
              />
          </Box>

          <Box className={"itemBox"}>
            <CustomPasswordField
              label={"Confirmar Senha"}
              name={"password2"}
              control={control}
              defaultValue=""
            />
          </Box>

          {/* <a href="http://127.0.0.1:8000/accounts/google/login">
            Cadastrar com Google
          </a> */}

          <div className="flex items-center justify-center p-3 space-x-32">
            {/* <CustomButton
                            type={"submit"}
                            label={"Register"}
                        /> */}
            <ReactiveButton
              className="font-bold text-primary hover:bg-yellow-300"
              buttonState={buttonState}
              color="yellow"
              type={"submit"}
              loadingText="Loading..."
              successText={"Cadastrador com sucesso"}
              idleText={"Cadastrar"}
              animation="true"
              rounded="true"
              width={80}
              height={40}
              size="large"
            />
          </div>
        </form>
      ) : (
        <p className="text-red-500">Link de convite inválido ou expirado.</p>
      )}
    </div>
  );
};

export default InviteRegister;
