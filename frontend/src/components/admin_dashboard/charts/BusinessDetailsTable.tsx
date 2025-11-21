import Box from "@mui/material/Box";
import { DataGrid } from "@mui/x-data-grid";
import { IconButton } from "@mui/material";
import { MdEdit } from "react-icons/md";
import { useMemo, useState } from "react";
import AxiosInstance from "../../AxiosInstance";
import BusinessDetailEdit from "../../BusinessDetailEdit";
import CustomLoadingOverlay, {
  CustomNoRowsOverlay,
} from "./components/CustomLoadingOverlay";

export default function BusinessDetailsTable({
  business_detail_data,
  onBusinessDetailUpdate,
  isLoading,
}) {
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [selectedBusinessDetail, setSelectedBusinessDetail] = useState(null);

  const handleEditClick = (businessDetail) => {
    console.log("BusinessDetail selected: ", businessDetail);
    setSelectedBusinessDetail(businessDetail);
    setEditModalOpen(true);
  };

  const handleSave = async (updatedBusinessDetail) => {
    console.log("Updated BusinessDetail: ", updatedBusinessDetail);
    try {
      const response = await AxiosInstance.put(
        `api/business-detail/${updatedBusinessDetail.id}/`,
        {
          id: updatedBusinessDetail.id,
          key: updatedBusinessDetail.key,
          value: updatedBusinessDetail.value,
        }
      );
      console.log(
        "BusinessDetail updated successfully: ",
        updatedBusinessDetail
      );

      if (response.status === 201 || response.status === 200) {
        onBusinessDetailUpdate();
      }
    } catch (error) {
      console.error("Error updating BusinessDetail: ", error);
    }
  };

  const handleRefreshTable = () => {
    console.log("Entered to refresh table");
    onBusinessDetailUpdate();
  };

  const handleClose = () => {
    // setLoading(false);
    setEditModalOpen(false);
  };

  const columns = useMemo(
    () => [
      {
        field: "id",
        headerName: <strong className="text-base">ID</strong>,
        width: 90,
        minWidth: 80,
      },
      {
        field: "key",
        headerName: <strong className="text-base">Título</strong>,
        minWidth: 200,
        flex: 2,
      },
      {
        field: "value",
        headerName: <strong className="text-base">Descrição</strong>,
        minWidth: 150,
        flex: 1,
      },
      {
        field: "actions",
        headerName: <strong className="text-base">Ações</strong>,
        width: 50,
        flex: 0.5,
        sortable: false,
        renderCell: (params) => (
          <IconButton
            onClick={() => handleEditClick(params.row)}
            size="small"
            color="primary"
          >
            <MdEdit color="#5D76D3" />
          </IconButton>
        ),
      },
    ],
    []
  );

  // Memoize rows so they don't change identity every render
  const rows = useMemo(() => business_detail_data, [business_detail_data]);

  return (
    <div className="flex w-full h-[689px]">
      <Box
        sx={{
          position: "relative",
          display: "flex",
          flexDirection: "column",
          width: "100%",
          pl: 1.5,
          pr: 1,
          py: 1,
          bottom: 0,
          border: "1px solid",
          borderColor: "divider",
          borderRadius: 1,
          "& .MuiDataGrid-root": {
            minWidth: 100, // Minimum width before scrolling
          },
        }}
      >
        <DataGrid
          rows={rows}
          columns={columns}
          getRowId={(row) => row.id}
          components={{
            LoadingOverlay: CustomLoadingOverlay,
            NoRowsOverlay: CustomNoRowsOverlay,
          }}
          slots={{
            noRowsOverlay: CustomNoRowsOverlay,
          }}
          slotProps={{
            noRowsOverlay: {
              // pass a dynamic message based on your app state
              content: isLoading
                ? "Carregando..."
                : "Nenhum serviço cadastrado.",
            },
          }}
          componentsProps={{
            noRowsOverlay: {
              // pass a dynamic message based on your app state
              message: isLoading
                ? "Carregando..."
                : "Nenhum Serviço cadastrado.",
            },
          }}
          loading={!!isLoading}
          // re-enable virtualization (remove disableVirtualization)
          disableSelectionOnClick
          // pageSizeOptions={[10, 15, 30]}
          initialState={{ pagination: { paginationModel: { pageSize: 15 } } }}
          density="comfortable"
          // important: set fixed row/header heights that match your styling
          rowHeight={28} // tune to match your desired visual height
          headerHeight={30}
          sx={{
            fontSize: "14px",
            boxShadow: 2,
            border: 2,
            borderColor: "var(--color-quaternary)",
            backgroundColor: "#FFFFFFFF",
            "& .MuiDataGrid-cell": {
              backgroundColor: "#F4F4F4FF",
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
              lineHeight: "1.2",
              padding: "8px",
            },
            "& .MuiDataGrid-columnHeaderTitle": {
              fontWeight: "bold",
            },
            "& .MuiDataGrid-cell:hover": {
              color: "var(--color-quaternary)",
            },
          }}
        />
        <BusinessDetailEdit
          open={editModalOpen}
          onClose={handleClose}
          businessDetail={selectedBusinessDetail}
          onSave={handleSave}
          refreshTable={handleRefreshTable}
        />
      </Box>
    </div>
  );
}
