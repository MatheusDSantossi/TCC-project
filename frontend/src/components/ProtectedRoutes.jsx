import { Outlet, Navigate } from "react-router-dom";
import { routes } from "../routes/routes";

const ProtectedRoute = () => {
    const adminHomeRoute = routes.admin.adminHome

    // Check if user is authenticated
    // If not, redirect to login page
    const token = localStorage.getItem('Token');
    // Check the expeiry of the token to make it more securitly
    return (
        token ? (
            <Outlet />
        ) : (
            <Navigate to={adminHomeRoute} />
        )
    )
};

export default ProtectedRoute;
