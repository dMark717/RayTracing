//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Dorozsami Márk
// Neptun : F5SXE8
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

enum MaterialType { ROUGH, REFLECTIVE, REFRACTIVE };

vec3 operator/(vec3 numerator, vec3 denominator) { return vec3(numerator.x / denominator.x, numerator.y / denominator.y, numerator.z / denominator.z); }

struct Material
{
	vec3 ka, kd, ks; // ambient, diffuse, specular
	float shininess;
	vec3 F0;
	float ior;
	MaterialType type;
	Material(MaterialType t) { type = t; }
};

struct RoughMaterial : Material
{
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH)
	{
		ka = _kd * 3;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}
};

struct ReflectiveMaterial : Material
{
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE)
	{
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	}
};

struct RefractiveMaterial : Material
{
	RefractiveMaterial(vec3 n) : Material(REFRACTIVE)
	{
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one)) / ((n + one) * (n + one));
		ior = n.x;
	}
};


struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

struct Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};


class Cylinder : public Intersectable {
	vec3 basePoint;
	vec3 axisDirection;
	float radius;
	float height;

public:
	Cylinder(vec3 _basePoint, vec3 _axisDirection, float _radius, float _height, Material* _material)
	{
		basePoint = _basePoint;
		axisDirection = normalize(_axisDirection);
		radius = _radius;
		height = _height;
		material = _material;
	}

	Hit intersect(const Ray& ray) override {

		Hit hit;

		vec3 oc = ray.start - basePoint;
		float a = dot(ray.dir - axisDirection * dot(ray.dir, axisDirection), ray.dir - axisDirection * dot(ray.dir, axisDirection));
		float b = 2 * dot(ray.dir - axisDirection * dot(ray.dir, axisDirection), oc - axisDirection * dot(oc, axisDirection));
		float c = dot(oc - axisDirection * dot(oc, axisDirection), oc - axisDirection * dot(oc, axisDirection)) - radius * radius;

		float discriminant = b * b - 4 * a * c;

		if (discriminant >= 0) {
			float t1 = (-b + sqrt(discriminant)) / (2 * a);
			float t2 = (-b - sqrt(discriminant)) / (2 * a);

			float t_min = fmin(t1, t2);
			float t_max = fmax(t1, t2);

			float t_hit = (t_min > 0 && t_max > 0) ? t_min : (t_max > 0 ? t_max : -1);

			if (t_hit > 0) {
				vec3 intersection_point = ray.start + ray.dir * t_hit;
				float projection = dot(intersection_point - basePoint, axisDirection);

				if (projection >= 0 && projection <= height) {
					hit.t = t_hit;
					hit.position = intersection_point;
					hit.normal = normalize(intersection_point - (basePoint + axisDirection * projection));
					hit.material = material;
				}
			}
		}
		return hit;
	}
};

class Cone : public Intersectable {
	vec3 apex;
	vec3 axisDirection;
	float angle;
	float height;

public:
	Cone(vec3 _apex, vec3 _axisDirection, float _angle, float _height, Material* _material)
	{
		apex = _apex;
		axisDirection = normalize(_axisDirection);
		angle = _angle;
		height = _height;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;

		vec3 oc = ray.start - apex;
		float cos_angle = cos(angle);

		float a = dot(ray.dir, axisDirection) * dot(ray.dir, axisDirection) - dot(ray.dir, ray.dir) * cos_angle * cos_angle;
		float b = 2 * (dot(ray.dir, axisDirection) * dot(oc, axisDirection) - dot(ray.dir, oc) * cos_angle * cos_angle);
		float c = dot(oc, axisDirection) * dot(oc, axisDirection) - dot(oc, oc) * cos_angle * cos_angle;

		float discriminant = b * b - 4 * a * c;

		if (discriminant >= 0) {
			float t1 = (-b + sqrt(discriminant)) / (2 * a);
			float t2 = (-b - sqrt(discriminant)) / (2 * a);

			float t_min = fmin(t1, t2);
			float t_max = fmax(t1, t2);

			float t_hit = (t_min > 0 && t_max > 0) ? t_min : (t_max > 0 ? t_max : -1);

			if (t_hit > 0) {
				vec3 intersection_point = ray.start + ray.dir * t_hit;
				float projection = dot(intersection_point - apex, axisDirection);

				if (projection >= 0 && projection <= height) {
					hit.t = t_hit;
					hit.position = intersection_point;
					hit.normal = normalize((apex + axisDirection * projection - intersection_point) / (cos_angle * (height - projection)));
					hit.material = material;
				}
			}
		}
		return hit;
	}

};


class CheckeredSquareGrid : public Intersectable {
	int gridSize;
	float squareSize;
	vec3 center;
	Material* material1;
	Material* material2;
public:
	CheckeredSquareGrid(int _gridSize, float _squareSize, vec3 _center, Material* _material1, Material* _material2) {
		gridSize = _gridSize;
		squareSize = _squareSize;
		center = _center;
		material1 = _material1;
		material2 = _material2;
	}

	Hit intersect(const Ray& ray) override {
		Hit hit;
		vec3 normal(0, 1, 0);
		float denom = dot(normal, ray.dir);
		if (denom == 0) return hit;

		float t = dot(center - ray.start, normal) / denom;
		if (t <= 0) return hit;

		vec3 intersectionPoint = ray.start + ray.dir * t;
		float halfSize = (gridSize * squareSize) / 2;

		if (intersectionPoint.x >= center.x - halfSize && intersectionPoint.x <= center.x + halfSize &&
			intersectionPoint.z >= center.z - halfSize && intersectionPoint.z <= center.z + halfSize) {
			hit.t = t;
			hit.position = intersectionPoint;
			hit.normal = normal;

			int row = (int)((intersectionPoint.x - (center.x - halfSize)) / squareSize);
			int col = (int)((intersectionPoint.z - (center.z - halfSize)) / squareSize);
			if ((row + col) % 2 == 0)
				hit.material = material1;
			else
				hit.material = material2;
		}
		return hit;
	}
};


class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float windowSize = length(w) * tanf(fov / 2);
		right = normalize(cross(vup, w)) * windowSize;
		up = normalize(cross(w, right)) * windowSize;
	}

	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2 * (X + 0.5f) / windowWidth - 1) + up * (2 * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}

	void Animate(float dt) {
		vec3 d = eye - lookat;
		eye = vec3(d.x * cos(dt) + d.z * sin(dt), d.y, -d.x * sin(dt) + d.z * cos(dt)) + lookat;
		vec3 up(0, 1, 0);
		set(eye, lookat, up, fov);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) { direction = normalize(_direction); Le = _Le; }
};

float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;

public:
	void build() {
		vec3 eye = vec3(0, 1, 4);
		vec3 vup = vec3(0, 1, 0);
		vec3 lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(1, 1, 1), Le(2, 2, 2);
		lights.push_back(new Light(lightDirection, Le));

		vec3 ks(2, 2, 2);

		Material* blueMaterial = new RoughMaterial(vec3(0, 0, 0.3), vec3(0, 0.1, 0.3), 100);
		Material* whiteMaterial = new RoughMaterial(vec3(0.3, 0.3, 0.3), vec3(0, 0, 0), 100);
		objects.push_back(new CheckeredSquareGrid(20, 1.0f, vec3(0, -1, 0), whiteMaterial, blueMaterial));

		Material* goldMaterial = new ReflectiveMaterial(vec3(0.17, 0.35, 1.5), vec3(3.1, 2.7, 1.9));
		objects.push_back(new Cylinder(vec3(1, -1, 0), vec3(0.1, 1, 0), 0.3f, 2.0f, goldMaterial));

		Material* waterRefractiveMaterial = new RefractiveMaterial(vec3(1.3, 1.3, 1.3));
		objects.push_back(new Cylinder(vec3(0, -1, -0.8), vec3(-0.2, 1, -0.1), 0.3f - 0.01f, 2.0f - 0.01f, waterRefractiveMaterial));

		Material* yellowPlasticMaterial = new RoughMaterial(vec3(0.3, 0.2, 0.1), vec3(2, 2, 2), 50);
		objects.push_back(new Cylinder(vec3(-1, -1, 0), vec3(0, 1, 0.1), 0.3f, 2.0f, yellowPlasticMaterial));

		Material* magentaMaterial = new RoughMaterial(vec3(0.3, 0, 0.2), vec3(2, 2, 2), 8);
		objects.push_back(new Cone(vec3(0, 1, 0), vec3(-0.1, -1, -0.05), 0.2f, 2.0f,magentaMaterial));

		Material* silverMaterial = new ReflectiveMaterial(vec3(0.14, 0.16, 0.13), vec3(4.1, 2.3, 3.1));
		objects.push_back(new Cone(vec3(0, 1, 0.8), vec3(0.2, -1, 0), 0.2f, 2.0f, silverMaterial));




	}

	void render(std::vector<vec4>& image) {

		for (int Y = 0; Y < windowHeight; Y++)
		{
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++)
			{
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(const Ray& ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		if (depth > 4) return La;
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;


		if (hit.material->type == ROUGH)
		{
			vec3 outRadiance = hit.material->ka * La;
			for (Light* light : lights) {
				Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
				float cosTheta = dot(hit.normal, light->direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
					outRadiance = outRadiance + hit.material->kd * light->Le * cosTheta;
					vec3 halfway = normalize(-ray.dir + light->direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + hit.material->ks * light->Le * powf(cosDelta, hit.material->shininess);
				}
			}
			return outRadiance;
		}

		float cosa = -dot(ray.dir, hit.normal);
		vec3 one(1, 1, 1);
		vec3 F = hit.material->F0 + (one - hit.material->F0) * powf(1 - cosa, 5);
		vec3 reflectDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
		vec3 outRadiance = trace(Ray(hit.position + hit.normal * epsilon, reflectDir), depth + 1) * F;

		if (hit.material->type == REFRACTIVE)
		{
			float disc = 1 - (1 - cosa * cosa) / hit.material->ior / hit.material->ior;
			if (disc >= 0) {
				vec3 refractedDir = ray.dir / hit.material->ior + hit.normal * (cosa / hit.material->ior - sqrt(disc));
				outRadiance = outRadiance +
					trace(Ray(hit.position - hit.normal * epsilon, refractedDir), depth + 1) * (one - F);
			}
		}
		return outRadiance;
	}

	void Animate(float dt) {
		camera.Animate(dt);
	}
};

Scene scene;
GPUProgram gpuProgram;


const char* const vertexSource = R"(
	#version 330
	precision highp float;		// normal floats, makes no difference on desktop computers

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 textcoord;

	void main() {
		textcoord = (cVertexPosition + vec2(1, 1)) / 2;	                // -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1);	// transform to clipping space
	}
)";

const char* const fragmentSource = R"(
	#version 330
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform sampler2D textureUnit;
	in vec2 textcoord; 	 // interpolated texture coordinates
	out vec4 fragmentColor;	 // output goes to the raster memory as told by glBindFragDataLocation

	void main() { fragmentColor = texture(textureUnit, textcoord); }
)";

class FullScreenTexturedQuad {
	unsigned int vao = 0, textureId = 0;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight) {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertices[] = { -1, -1, 1, -1, 1, 1, -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	void LoadTexture(std::vector<vec4>& image) {
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);
	}

	void Draw()
	{
		glBindVertexArray(vao);
		int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
		const unsigned int textureUnit = 0;
		if (location >= 0) {
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureId);
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

void onInitialization()
{
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight);
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay()
{
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad->LoadTexture(image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'a') scene.Animate(2 * M_PI / 8);
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

void onMouse(int button, int state, int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char* buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

void onIdle() {
	scene.Animate(0.05f);
	glutPostRedisplay();
}