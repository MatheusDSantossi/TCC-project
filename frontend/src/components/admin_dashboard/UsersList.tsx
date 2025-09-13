import { useEffect, useState, useMemo } from "react";
import AxiosInstance from "../AxiosInstance";
import { Box } from "@mui/material";

const UserList = () => {
  const [users, setUsers] = useState();
  const [loading, setLoading] = useState(true);

  const GetData = () => {
    AxiosInstance.get(`users/`).then((response) => {
      setUsers(response.data);
      console.log(response.data);
      setLoading(false);
    });
  };

  useEffect(() => {
    GetData();
  }, []);

  console.log("users: ", users);
  //   I'm not using this component

  return (
    <div>
      {loading ? (
        <p>Loading data...</p>
      ) : (
        <div>
          {users.map((user, index) => (
            <Box key={index} sx={{ p: 2, m: 2, boxShadow: 3 }}>
              <div className="font-medium text-primary">ID: {user.id}</div>
              <div className="font-medium text-primary">
                Email: {user.email}
              </div>
            </Box>
          ))}
        </div>
      )}
    </div>
  );
};

export default UserList;
