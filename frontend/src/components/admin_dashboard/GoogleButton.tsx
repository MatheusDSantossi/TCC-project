import { useEffect } from "react";
import AxiosInstance from "../AxiosInstance";
import { useNavigate } from "react-router-dom";

const GoogleButton = () => {
  const navigate = useNavigate();

  const handleGoogleLogin = () => {
    window.location.href = "http://localhost:8000/accounts/google/login";
  };

  useEffect(() => {
    const handleCallback = async () => {
      const params = new URLSearchParams(window.location.search);
      const code = params.get("code");
      const error = params.get("error");

      if (error) {
        console.error("Google login failed:", error);
      }

      if (code) {
        try {
          const response = await AxiosInstance.post(
            "/api/auth/google/callback/",
            { code }
          );
          localStorage.setItem("Token", response.data.token);
          navigate("/dashboard");
        } catch (err) {
          console.error("Callback error: ", err);
        }
      }
    };
    handleCallback();
  }, [navigate]);

  return <button onClick={handleGoogleLogin}>Login with Google</button>;
};

export default GoogleButton;
