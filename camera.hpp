
// Slightly modified version of https://github.com/SaschaWillems/Vulkan camera class
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera
{
	private:
		float fov = 0.f;
		float znear =  0.f, zfar = 0.f;

		void updateViewMatrix()
		{
			glm::mat4 rotM = glm::mat4(1.0f);
			glm::mat4 transM;

			rotM = glm::rotate(rotM, glm::radians(rotation.x * (flipY ? -1.0f : 1.0f)), glm::vec3(1.0f, 0.0f, 0.0f));
			rotM = glm::rotate(rotM, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
			//rotM = glm::rotate(rotM, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

			glm::vec3 translation = position;
			if (flipY) {
				translation.y *= -1.0f;
			}
			transM = glm::translate(glm::mat4(1.0f), translation);

			if (type == CameraType::firstperson)
			{
				matrices.view = rotM * transM;
			}
			else
			{
				matrices.view = transM * rotM;
			}

			viewPos = glm::vec4(position, 0.0f) * glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f);

			updated = true;

			glm::mat4 viewProj =  glm::transpose(matrices.perspective * matrices.view);

			planes[0] = viewProj[3] + viewProj[0];
			// right
			planes[1] = viewProj[3] - viewProj[0];
			// top
			planes[2] = viewProj[3] - viewProj[1];
			// bottom
			planes[3] = viewProj[3] + viewProj[1];
			// near
			planes[4] = viewProj[3] + viewProj[2];
			// far
			planes[5] = viewProj[3] - viewProj[2];

			for (int i = 0; i < 6; i++)
			{
				float length = sqrt((planes[i].x * planes[i].x) + (planes[i].y * planes[i].y) + (planes[i].z * planes[i].z));
				planes[i].x /= length;
				planes[i].y /= length;
				planes[i].z /= length;
				planes[i].w /= length;
			}

		}
	
	public:
		enum class CameraType { lookat, firstperson};
		CameraType type = CameraType::firstperson;
		glm::vec3 rotation = glm::vec3();
		glm::vec3 position = glm::vec3();
		glm::vec4 viewPos = glm::vec4();

		float rotationSpeed = 1.0f;
		float movementSpeed = 1.0f;

		bool updated = false;
		bool flipY = false;
		// frustum planes
		glm::vec4 planes[6];

		struct
		{
			glm::mat4 perspective;
			glm::mat4 view;
		} matrices;

		struct
		{
			bool left = false;
			bool right = false;
			bool up = false;
			bool down = false;
		} keys;

		bool moving()
		{
			return keys.left || keys.right || keys.up || keys.down;
		}

		float getNearClip() {
			return znear;
		}

		float getFarClip() {
			return zfar;
		}

		void setPerspective(float fov, float aspect, float znear, float zfar)
		{
			this->fov = fov;
			this->znear = znear;
			this->zfar = zfar;
			matrices.perspective = glm::perspective(glm::radians(fov), aspect, znear, zfar);
			if (flipY) {
				matrices.perspective[1, 1] *= -1.0f;
			}
		};

		void updateAspectRatio(float aspect)
		{
			matrices.perspective = glm::perspective(glm::radians(fov), aspect, znear, zfar);
			if (flipY) {
				matrices.perspective[1, 1] *= -1.0f;
			}
		}

		void setPosition(glm::vec3 position)
		{
			this->position = position;
			updateViewMatrix();
		}

		void setRotation(glm::vec3 rotation)
		{
			this->rotation = rotation;
			updateViewMatrix();
		}

		void rotate(glm::vec3 delta)
		{
			this->rotation += delta;
			updateViewMatrix();
		}

		void setTranslation(glm::vec3 translation)
		{
			this->position = translation;
			updateViewMatrix();
		};

		void translate(glm::vec3 delta)
		{
			this->position += delta;
			updateViewMatrix();
		}

		void setRotationSpeed(float rotationSpeed)
		{
			this->rotationSpeed = rotationSpeed;
		}

		void setMovementSpeed(float movementSpeed)
		{
			this->movementSpeed = movementSpeed;
		}

		void update(float deltaTime)
		{
			updated = false;
			if (type == CameraType::firstperson)
			{
				if (moving())
				{
					glm::vec3 camFront;
					float x = glm::radians(rotation.x); // pitch ?
					float y = glm::radians(rotation.y); // yaw ?

					camFront.x = -cos(x) * sin(y);
					camFront.y = sin(x);
					camFront.z = cos(x) * cos(y);
					camFront = glm::normalize(camFront);

					float moveSpeed = deltaTime * movementSpeed;

					if (keys.up)
						position += camFront * moveSpeed;
					if (keys.down)
						position -= camFront * moveSpeed;
					if (keys.left)
						position -= glm::normalize(glm::cross(camFront, glm::vec3(0.0f, 1.0f, 0.0f))) * moveSpeed;
					if (keys.right)
						position += glm::normalize(glm::cross(camFront, glm::vec3(0.0f, 1.0f, 0.0f))) * moveSpeed;

					updateViewMatrix();
				}
			}
		};

};